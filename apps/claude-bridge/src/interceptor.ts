import fs from "fs";
import path from "path";
import {
	RawPair,
	BridgeConfig,
	TransformationEntry,
	Provider,
	ProviderClientInfo,
	CapabilityValidationResult,
	JSONSchema,
} from "./types.js";
import { transformAnthropicToLemmy } from "./transforms/anthropic-to-lemmy.js";
import { createAnthropicSSE } from "./transforms/lemmy-to-anthropic.js";
import { jsonSchemaToZod } from "./transforms/tool-schemas.js";
import type { MessageCreateParamsBase } from "@anthropic-ai/sdk/resources/messages/messages.js";
import {
	Context,
	type AskResult,
	type ToolDefinition,
	type SerializedContext,
	type SerializedToolDefinition,
	type Message,
	type ToolResult,
	type Attachment,
	AskInput,
} from "@mariozechner/lemmy";
import { lemmy } from "@mariozechner/lemmy";
import { z } from "zod";
import { FileLogger, NullLogger, type Logger } from "./utils/logger.js";
import { parseSSE, extractAssistantFromSSE } from "./utils/sse.js";
import {
	parseAnthropicMessageCreateRequest,
	parseResponse,
	isAnthropicAPI,
	generateRequestId,
	type ParsedRequestData,
} from "./utils/request-parser.js";
import { createProviderClient, validateCapabilities, convertThinkingParameters } from "./utils/provider.js";

// We'll rely on config.toolsEnabled setting instead of model-specific compatibility

export class ClaudeBridgeInterceptor {
	private config!: BridgeConfig;
	private logger!: Logger;
	private requestsFile!: string;
	private transformedFile!: string;
	private contextFile!: string;
	private traceFile!: string;
	private clientInfo!: ProviderClientInfo;
	private pendingRequests = new Map<string, any>();

	/**
	 * Create a new interceptor instance (async factory)
	 */
	static async create(config: BridgeConfig): Promise<ClaudeBridgeInterceptor> {
		const instance = new ClaudeBridgeInterceptor();
		await instance.initialize(config);
		return instance;
	}

	private constructor() {
		// Private constructor - use create() instead
	}

	private async initialize(config: BridgeConfig): Promise<void> {
		this.config = { logDirectory: ".claude-bridge", logLevel: "info", debug: false, ...config };

		// Trace mode implies debug mode
		if (this.config.trace) {
			this.config.debug = true;
		}

		// Setup logging based on debug flag
		if (this.config.debug) {
			const logDir = this.config.logDirectory!;
			if (!fs.existsSync(logDir)) fs.mkdirSync(logDir, { recursive: true });
			this.logger = new FileLogger(logDir);

			// Setup files
			const timestamp = new Date().toISOString().replace(/[:.]/g, "-").replace("T", "-").slice(0, -5);
			this.requestsFile = path.join(logDir, `requests-${timestamp}.jsonl`);
			this.transformedFile = path.join(logDir, `transformed-${timestamp}.jsonl`);
			this.contextFile = path.join(logDir, `context-${timestamp}.jsonl`);
			this.traceFile = path.join(logDir, `trace-${timestamp}.jsonl`);
			fs.writeFileSync(this.requestsFile, "");
			fs.writeFileSync(this.transformedFile, "");
			fs.writeFileSync(this.contextFile, "");
			fs.writeFileSync(this.traceFile, "");
		} else {
			this.logger = new NullLogger();
			// Set dummy file paths when not logging
			this.requestsFile = "";
			this.transformedFile = "";
			this.contextFile = "";
			this.traceFile = "";
		}

		// Setup provider-agnostic client
		this.clientInfo = await createProviderClient(this.config);

		this.logger.log(`Requests logged to ${this.requestsFile}`);
		this.logger.log(`Transformed requests logged to ${this.transformedFile}`);
		this.logger.log(`Initialized ${this.clientInfo.provider} client for model: ${this.clientInfo.model}`);
	}

	public instrumentFetch(): void {
		if (!global.fetch || (global.fetch as any).__claudeBridgeInstrumented) return;

		const originalFetch = global.fetch;
		global.fetch = async (input: Parameters<typeof fetch>[0], init: RequestInit = {}): Promise<Response> => {
			const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
			if (!isAnthropicAPI(url)) return originalFetch(input, init);
			return this.handleAnthropicRequest(originalFetch, input, init);
		};

		(global.fetch as any).__claudeBridgeInstrumented = true;
		this.logger.log("Claude Bridge interceptor initialized");
	}

	private async handleAnthropicRequest(
		originalFetch: typeof fetch,
		input: Parameters<typeof fetch>[0],
		init: RequestInit,
	): Promise<Response> {
		const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
		this.logger.log(`Intercepted Claude request: ${url}`);

		const requestId = generateRequestId();
		const requestData = await parseAnthropicMessageCreateRequest(url, init, this.logger);

		// Detect problematic message patterns for OpenAI compatibility
		this.detectProblematicMessagePatterns(requestData);

		const transformResult = await this.tryTransform(requestData);
		this.pendingRequests.set(requestId, requestData);

		// Log trace information if in trace mode
		if (this.config.trace && requestData.body) {
			const anthropicRequest = requestData.body as MessageCreateParamsBase;
			const traceEntry = {
				timestamp: new Date().toISOString(),
				model: anthropicRequest.model,
				system_prompt: anthropicRequest.system || null,
				tools: anthropicRequest.tools || null,
				thinking_enabled: !!anthropicRequest.thinking,
				max_tokens: anthropicRequest.max_tokens,
				temperature: anthropicRequest.temperature,
				messages: transformResult ? transformResult.messages : anthropicRequest.messages,
				...(transformResult && { serialized_context: transformResult }),
			};
			fs.appendFileSync(this.traceFile, JSON.stringify(traceEntry) + "\n");
		}

		try {
			// In trace mode, always call original Anthropic API (no transformation)
			// Get response from provider or fallback to Anthropic
			const modifiedRequestData = { ...requestData };

			// Always ensure tools are properly handled for all providers
			if (modifiedRequestData.body) {
				// Log original request content
				this.logger.log(
					`📥 Original request content: ${JSON.stringify(modifiedRequestData.body, null, 2).substring(0, 500)}${JSON.stringify(modifiedRequestData.body, null, 2).length > 500 ? "..." : ""}`,
				);

				// Deep clone the body to avoid modifying original request
				const body = { ...(modifiedRequestData.body as MessageCreateParamsBase) };

				// Log the original request's tools if present
				if (body.tools) {
					this.logger.log(`📝 Original request contains ${body.tools.length} tools`);

					// Log each tool for debugging
					body.tools.forEach((tool, index) => {
						this.logger.log(`📋 Tool[${index}]: ${tool.name} (${tool.type || "custom"})`);
					});
				}

				// Remove tools completely if tools are disabled
				if (!this.config.toolsEnabled) {
					this.logger.log(`🔧 Tools disabled - removing tools from request`);
					delete body.tools;
				}

				modifiedRequestData.body = body;

				// Log the modified request after tool handling
				this.logger.log(
					`🔄 Final request ${modifiedRequestData.body.tools ? "contains" : "does not contain"} tools`,
				);
				this.logger.log(
					`📤 Modified request content: ${JSON.stringify(modifiedRequestData.body, null, 2).substring(0, 500)}${JSON.stringify(modifiedRequestData.body, null, 2).length > 500 ? "..." : ""}`,
				);
			}

			const response =
				this.config.trace || !transformResult
					? await originalFetch(input, init)
					: await this.callProvider(transformResult, modifiedRequestData.body);

			// Log response status
			this.logger.log(`📩 Response status: ${response.status} ${response.statusText}`);

			// Log everything
			await this.logComplete(modifiedRequestData, response, transformResult, requestId);

			this.pendingRequests.delete(requestId);
			return response;
		} catch (error) {
			this.pendingRequests.delete(requestId);
			throw error;
		}
	}

	private async tryTransform(requestData: ParsedRequestData): Promise<SerializedContext | null> {
		try {
			if (requestData.method !== "POST" || !requestData.body) return null;

			const anthropicRequest = requestData.body as MessageCreateParamsBase;

			// Skip haiku models (except in trace mode)
			if (!this.config.trace && anthropicRequest.model?.toLowerCase().includes("haiku")) {
				this.logger.log(`Skipping transformation for haiku model: ${anthropicRequest.model}`);
				return null;
			}

			return transformAnthropicToLemmy(anthropicRequest);
		} catch (error) {
			if (error instanceof Error && error.message.includes("Multi-turn conversations")) {
				this.logger.log(`Skipping transformation for multi-turn conversation: ${error.message}`);
				return null;
			}
			this.logger.error(`Failed to transform request: ${error instanceof Error ? error.message : String(error)}`);
			return null;
		}
	}

	private async callProvider(
		transformResult: SerializedContext,
		originalRequest: MessageCreateParamsBase,
	): Promise<Response> {
		try {
			// We now rely on the toolsEnabled config from CLI arguments

			// Validate capabilities (skip for unknown models)
			let validation: CapabilityValidationResult = { valid: true, warnings: [], adjustments: {} };
			if (this.clientInfo.modelData) {
				validation = validateCapabilities(this.clientInfo.modelData, originalRequest, this.logger);
				if (!validation.valid) {
					validation.warnings.forEach((warning: string) => this.logger.log(`⚠️  ${warning}`));
				}
			} else {
				this.logger.log(`⚠️  Skipping capability validation for unknown model: ${this.clientInfo.model}`);
			}

			this.logger.log(`🔍 Tools Enabled: ${this.config.toolsEnabled}`);
			this.logger.log(`🔍 Total Tools Before Filtering: ${transformResult.tools.length}`);

			// Create dummy tools for deserialization
			const dummyTools: ToolDefinition[] = [];

			// CRITICAL: If tools are disabled, remove them from the serialized context
			// This prevents "Cannot restore tool" errors when deserializing
			if (!this.config.toolsEnabled) {
				this.logger.log(`🔧 Tools disabled - removing tools from serialized context`);
				// Make a deep copy of the context to avoid modifying the original
				transformResult = {
					...transformResult,
					tools: [], // Empty the tools array
				};

				// Also remove any tool call references from messages
				transformResult.messages = transformResult.messages.map((msg) => {
					// For assistant messages, remove toolCalls
					if (msg.role === "assistant" && "toolCalls" in msg) {
						const { toolCalls, ...rest } = msg;
						return rest;
					}

					// For user messages, remove toolResults
					if (msg.role === "user" && "toolResults" in msg) {
						const { toolResults, ...rest } = msg;
						return rest;
					}

					return msg;
				});

				this.logger.log(`🔧 Removed all tool references from serialized context`);
			} else if (this.config.toolsEnabled === true) {
				// Only add tools if explicitly enabled
				this.logger.log(`🔧 Tools explicitly enabled - adding ${transformResult.tools.length} tool definitions`);
				transformResult.tools.forEach((tool: SerializedToolDefinition) => {
					try {
						dummyTools.push({
							name: tool.name,
							description: tool.description,
							schema: this.safeJsonSchemaToZod(tool.jsonSchema as JSONSchema),
							execute: async () => {
								throw new Error("Tool execution not supported in bridge mode");
							},
						});
					} catch (error) {
						this.logger.log(`⚠️ Failed to create tool '${tool.name}': ${error}`);
					}
				});
			} else {
				this.logger.log(`🔧 Tools disabled - not adding any tool definitions`);
			}

			this.logger.log(`🔍 Dummy Tools Count: ${dummyTools.length}`);

			// Deserialize context and call provider
			this.logger.log(
				`🔄 Deserializing context with ${dummyTools.length} tools and ${transformResult.tools.length} tool references`,
			);
			const context = Context.deserialize(transformResult, dummyTools);
			const lastMessage = context.getMessages()[context.getMessages().length - 1];

			// Construct proper AskInput from the last user message
			let askInput: AskInput | string = "";
			if (lastMessage?.role === "user") {
				const userMessage = lastMessage;
				askInput = {
					...(userMessage.content && { content: userMessage.content }),
					...(userMessage.toolResults && { toolResults: userMessage.toolResults }),
					...(userMessage.attachments && { attachments: userMessage.attachments }),
				};
			}

			// Convert thinking parameters for provider
			const askOptions = convertThinkingParameters(this.clientInfo.provider, originalRequest);

			// Apply capability adjustments
			if (validation.adjustments.maxOutputTokens) {
				askOptions.maxOutputTokens = validation.adjustments.maxOutputTokens;
			}

			this.logger.log(`Calling ${this.clientInfo.provider} with model: ${this.clientInfo.model}`);

			// Log the actual request being sent to the provider
			this.logger.log(`🔍 Provider request details:`);
			this.logger.log(`🔹 Provider: ${this.clientInfo.provider}`);
			this.logger.log(`🔹 Model: ${this.clientInfo.model}`);
			this.logger.log(
				`🔹 AskInput: ${typeof askInput === "string" ? askInput : JSON.stringify(askInput).substring(0, 200)}${typeof askInput !== "string" && JSON.stringify(askInput).length > 200 ? "..." : ""}`,
			);
			this.logger.log(
				`🔹 AskOptions: ${JSON.stringify(askOptions).substring(0, 200)}${JSON.stringify(askOptions).length > 200 ? "..." : ""}`,
			);

			const askResult: AskResult = await this.clientInfo.client.ask(askInput, { context, ...askOptions });

			if (askResult.type !== "success") {
				this.logger.error(`${this.clientInfo.provider} error response: ${JSON.stringify(askResult.error)}`);
				throw new Error(askResult.error?.message || JSON.stringify(askResult.error) || "Request failed");
			}

			// Convert to Anthropic SSE format
			return new Response(createAnthropicSSE(askResult, this.clientInfo.model), {
				status: 200,
				statusText: "OK",
				headers: {
					"Content-Type": "text/event-stream",
					"Cache-Control": "no-cache",
					Connection: "keep-alive",
					"anthropic-request-id": generateRequestId(),
				},
			});
		} catch (error) {
			this.logProviderError(error);
			throw error;
		}
	}

	private async logComplete(
		requestData: ParsedRequestData,
		response: Response,
		transformResult: SerializedContext | null,
		requestId: string,
	) {
		const responseData = await parseResponse(response.clone());

		// Log raw request-response pair
		const pair: RawPair = {
			request: requestData,
			response: responseData,
			logged_at: new Date().toISOString(),
		};
		if (this.config.debug) {
			fs.appendFileSync(this.requestsFile, JSON.stringify(pair) + "\n");
		}

		// Log transformation entry if we transformed
		if (transformResult) {
			const decodedSSE =
				responseData.body_raw && responseData.headers["content-type"]?.includes("text/event-stream")
					? parseSSE(responseData.body_raw)
					: undefined;

			const contextWithResponse = { ...transformResult };
			const assistantResponse = decodedSSE ? extractAssistantFromSSE(decodedSSE, this.logger) : null;
			if (assistantResponse) {
				contextWithResponse.messages = [...contextWithResponse.messages, assistantResponse];
			}

			// Log messages to context.jsonl
			if (this.config.debug) {
				const logEntry = {
					timestamp: new Date().toISOString(),
					messages: contextWithResponse.messages,
				};
				fs.appendFileSync(this.contextFile, JSON.stringify(logEntry) + "\n");
			}

			const transformEntry: TransformationEntry = {
				timestamp: Date.now() / 1000,
				request_id: requestId,
				raw_request: requestData.body as MessageCreateParamsBase,
				lemmy_context: contextWithResponse,
				bridge_config: { provider: this.config.provider || "unknown", model: this.config.model || "unknown" },
				raw_response: responseData,
				decoded_sse: decodedSSE,
				logged_at: new Date().toISOString(),
			};

			if (this.config.debug) {
				fs.appendFileSync(this.transformedFile, JSON.stringify(transformEntry) + "\n");
			}
			this.logger.log(`Transformed and logged request with response to ${this.transformedFile}`);
		}

		this.logger.log(`Logged request-response pair to ${this.requestsFile}`);
	}

	private safeJsonSchemaToZod(jsonSchema: JSONSchema): z.ZodSchema {
		try {
			return jsonSchemaToZod(jsonSchema);
		} catch {
			return z.any();
		}
	}

	private logProviderError(error: unknown) {
		this.logger.error(`CRITICAL: ${this.clientInfo.provider} request failed with detailed error information:`);
		this.logger.error(`Error type: ${typeof error}`);
		this.logger.error(`Error constructor: ${error?.constructor?.name}`);

		if (error instanceof Error) {
			this.logger.error(`Error message: ${error.message}`);
			this.logger.error(`Error stack: ${error.stack}`);
			if ("cause" in error && error.cause) this.logger.error(`Error cause: ${JSON.stringify(error.cause)}`);
			if ("code" in error) this.logger.error(`Error code: ${(error as any).code}`);
			if ("status" in error) this.logger.error(`HTTP status: ${(error as any).status}`);
		} else {
			this.logger.error(`Non-Error object: ${JSON.stringify(error, null, 2)}`);
		}

		this.logger.error(
			`Config: ${JSON.stringify({
				provider: this.config.provider,
				model: this.config.model,
				apiKey: this.config.apiKey ? `${this.config.apiKey.substring(0, 10)}...` : "NOT_SET",
			})}`,
		);
	}

	private detectProblematicMessagePatterns(requestData: ParsedRequestData): void {
		if (!requestData.body?.messages || !Array.isArray(requestData.body.messages)) {
			return;
		}

		const messages = requestData.body.messages as any[];

		for (let i = 0; i < messages.length - 1; i++) {
			const currentMessage = messages[i];
			const nextMessage = messages[i + 1];

			// Check for: assistant message with tool calls followed by user message without tool results
			if (
				currentMessage &&
				nextMessage &&
				currentMessage.role === "assistant" &&
				currentMessage.tool_calls &&
				Array.isArray(currentMessage.tool_calls) &&
				currentMessage.tool_calls.length > 0 &&
				nextMessage.role === "user" &&
				!nextMessage.tool_call_id &&
				!nextMessage.tool_result_id
			) {
				this.logger.log(
					`🚨 DETECTED PROBLEMATIC PATTERN: Assistant message with ${currentMessage.tool_calls.length} tool calls (position ${i}) followed by user message without tool results (position ${i + 1})`,
				);
				this.logger.log(
					`Tool call IDs: ${currentMessage.tool_calls.map((tc: { id: string }) => tc.id).join(", ")}`,
				);
				this.logger.log(
					`User message content preview: ${typeof nextMessage.content === "string" ? nextMessage.content.substring(0, 100) : "[complex content]"}`,
				);
			}
		}
	}

	public cleanup(): void {
		this.logger.log("Cleaning up interceptor...");
		for (const [, requestData] of this.pendingRequests.entries()) {
			const orphaned = {
				request: requestData,
				response: null,
				note: "ORPHANED_REQUEST",
				logged_at: new Date().toISOString(),
			};
			if (this.config.debug) {
				fs.appendFileSync(this.requestsFile, JSON.stringify(orphaned) + "\n");
			}
		}
		this.pendingRequests.clear();
		this.logger.log(`Cleanup complete.`);
	}
}

// Global interceptor management
let globalInterceptor: ClaudeBridgeInterceptor | null = null;
let eventListenersSetup = false;

export async function initializeInterceptor(config?: BridgeConfig): Promise<ClaudeBridgeInterceptor> {
	if (globalInterceptor) {
		console.warn("⚠️  Interceptor already initialized");
		return globalInterceptor;
	}

	const defaultConfig: BridgeConfig = {
		provider: (process.env["CLAUDE_BRIDGE_PROVIDER"] as Provider) || "openai",
		model: process.env["CLAUDE_BRIDGE_MODEL"] || "gpt-4o",
		apiKey: process.env["CLAUDE_BRIDGE_API_KEY"],
		baseURL: process.env["CLAUDE_BRIDGE_BASE_URL"],
		maxRetries: process.env["CLAUDE_BRIDGE_MAX_RETRIES"]
			? parseInt(process.env["CLAUDE_BRIDGE_MAX_RETRIES"])
			: undefined,
		logDirectory: process.env["CLAUDE_BRIDGE_LOG_DIR"] || ".claude-bridge",
		debug: process.env["CLAUDE_BRIDGE_DEBUG"] === "true",
		trace: process.env["CLAUDE_BRIDGE_TRACE"] === "true",
		toolsEnabled:
			process.env["CLAUDE_BRIDGE_TOOLS_ENABLED"] === undefined
				? true
				: process.env["CLAUDE_BRIDGE_TOOLS_ENABLED"] === "true",
	};

	globalInterceptor = await ClaudeBridgeInterceptor.create({ ...defaultConfig, ...config });
	globalInterceptor.instrumentFetch();

	if (!eventListenersSetup) {
		const cleanup = () => globalInterceptor?.cleanup();
		process.on("exit", cleanup);
		process.on("SIGINT", cleanup);
		process.on("SIGTERM", cleanup);
		process.on("uncaughtException", (error) => {
			console.error("Uncaught exception:", error);
			cleanup();
			process.exit(1);
		});
		eventListenersSetup = true;
	}

	return globalInterceptor;
}

export function getInterceptor(): ClaudeBridgeInterceptor | null {
	return globalInterceptor;
}
