#!/usr/bin/env node

// Suppress deprecation warnings
process.removeAllListeners("warning");

import { spawnSync } from "child_process";
import {
	getDefaultApiKeyEnvVar,
	getProviders,
	AnthropicModelData,
	OpenAIModelData,
	GoogleModelData,
	ModelToProvider,
	findModelData,
	type AllModels,
	type ModelData,
	type Provider,
} from "@mariozechner/lemmy";
import path from "path";
import { fileURLToPath } from "url";
import { patchClaudeBinary } from "./patch-claude.js";
import { VERSION } from "./version.js";

interface ClaudeArgs {
	provider: Provider;
	model: string;
	apiKey?: string | undefined;
	baseURL?: string | undefined;
	maxRetries?: number | undefined;
	logDir?: string | undefined;
	patchClaude?: boolean | undefined;
	debug?: boolean | undefined;
	trace?: boolean | undefined;
	toolsEnabled?: boolean;
	claudeArgs: string[];
}

interface ParsedArgs {
	version?: boolean | undefined;
	help?: boolean | undefined;
	trace?: boolean | undefined;
	provider?: string | undefined;
	model?: string | undefined;
	apiKey?: string | undefined;
	baseURL?: string | undefined;
	maxRetries?: number | undefined;
	logDir?: string | undefined;
	patchClaude?: boolean | undefined;
	debug?: boolean | undefined;
	toolsEnabled?: boolean;
	claudeArgs: string[];
}

// Get models that support both tools and images - type-safe and exhaustive
function getCapableModels(): Record<Provider, string[]> {
	const capableModels: Record<Provider, string[]> = {
		anthropic: [],
		openai: [],
		google: [],
	};

	// Check Anthropic models
	for (const [model, data] of Object.entries(AnthropicModelData)) {
		if (data && data.supportsTools && data.supportsImageInput) {
			capableModels.anthropic.push(model);
		}
	}

	// Check OpenAI models
	for (const [model, data] of Object.entries(OpenAIModelData)) {
		if (data && data.supportsTools && data.supportsImageInput) {
			capableModels.openai.push(model);
		}
	}

	// Check Google models
	for (const [model, data] of Object.entries(GoogleModelData)) {
		if (data && data.supportsTools && data.supportsImageInput) {
			capableModels.google.push(model);
		}
	}

	return capableModels;
}

// Get capable models for a specific provider with exhaustive type checking
function getCapableModelsForProvider(provider: Provider): string[] {
	const allCapableModels = getCapableModels();

	switch (provider) {
		case "anthropic":
			return allCapableModels.anthropic;
		case "openai":
			return allCapableModels.openai;
		case "google":
			return allCapableModels.google;
		default:
			// TypeScript will catch if we miss any provider cases
			const _exhaustiveCheck: never = provider;
			return _exhaustiveCheck;
	}
}

// Validate provider with exhaustive type checking
function isValidProvider(provider: string): provider is Provider {
	switch (provider) {
		case "anthropic":
		case "openai":
		case "google":
			return true;
		default:
			return false;
	}
}

// Get all valid providers with exhaustive type checking
function getValidProviders(): Provider[] {
	return ["anthropic", "openai", "google"];
}

// Filter to only non-Anthropic providers (since we're bridging to non-Anthropic)
function getNonAnthropicProviders(): Exclude<Provider, "anthropic">[] {
	return ["openai", "google"];
}

function formatModelInfo(model: string, data: ModelData): string {
	const tools = data.supportsTools ? "✓" : "✗";
	const images = data.supportsImageInput ? "✓" : "✗";
	const maxInput = data.contextWindow.toLocaleString();
	const maxOutput = data.maxOutputTokens.toLocaleString();
	return `  ${model.padEnd(35)} ${tools.padStart(6)}  ${images.padStart(7)}  ${maxInput.padStart(12)}  ${maxOutput.padStart(12)}`;
}

function showHelp(): void {
	console.log(`claude-bridge - Use non-Anthropic models with Claude Code\nVersion: ${VERSION}

USAGE:
  claude-bridge                           Show all available providers
  claude-bridge <provider>                Show models for a provider
  claude-bridge <provider> <model>        Run with provider and model
  claude-bridge --trace <claude args>     Spy on Claude Code ↔ Anthropic communication
  claude-bridge --version                 Show version information
  claude-bridge --help                    Show this help

EXAMPLES:
  # Natural discovery flow
  claude-bridge                           # Shows: openai, google
  claude-bridge openai                    # Shows OpenAI models
  claude-bridge google                    # Shows Google models

  # Execution
  claude-bridge openai gpt-4o
  claude-bridge google gemini-2.0-flash-exp

  # With custom configuration
  claude-bridge openai gpt-4o --apiKey sk-... --baseURL https://api.openai.com/v1

  # Single-shot prompts
  claude-bridge openai gpt-4o -p "Hello world"
  claude-bridge google gemini-1.5-pro -p "Debug this code"

OPTIONS:
  --apiKey <key>        API key for the provider
  --baseURL <url>       Custom API base URL
  --maxRetries <num>    Maximum number of retries for failed requests
  --log-dir <dir>       Directory for log files (default: .claude-bridge)
  --patch-claude        Patch Claude binary to disable anti-debugging checks
  --debug               Enable debug logging (requests/responses to .claude-bridge/)
  --trace               Spy mode: log all Claude ↔ Anthropic communication (implies --debug)
  --tools-enable        Enable tool support for this session (default)
  --tools-disable       Disable tool support for this session
  --version             Show version information
  --help, -h            Show this help

ENVIRONMENT VARIABLES:
  OPENAI_API_KEY        API key for OpenAI (if --apiKey not provided)
  GOOGLE_API_KEY        API key for Google (if --apiKey not provided)

NOTE:
  Only models with both tools and image support are shown by default.
  Use --debug to enable request/response logging to .claude-bridge/
`);
}

function showProviders(): void {
	const nonAnthropicProviders = getNonAnthropicProviders();

	console.log(`Available providers (only showing providers with capable models):\n`);

	for (const provider of nonAnthropicProviders) {
		const models = getCapableModelsForProvider(provider);
		if (models.length > 0) {
			switch (provider) {
				case "openai":
					console.log(`  openai     OpenAI models (GPT-4o, etc.)`);
					break;
				case "google":
					console.log(`  google     Google models (Gemini, etc.)`);
					break;
				default:
					// TypeScript will catch if we miss any provider cases
					const _exhaustiveCheck: never = provider;
					_exhaustiveCheck;
			}
		}
	}

	console.log(`
Usage:
  claude-bridge <provider>        Show models for a provider
  claude-bridge --help            Show detailed help

Examples:
  claude-bridge openai           # Show OpenAI models
  claude-bridge google           # Show Google models`);
}

function showProviderModels(provider: string): void {
	// Validate provider first
	if (!isValidProvider(provider)) {
		console.error(`❌ Invalid provider: ${provider}`);
		const validProviders = getNonAnthropicProviders();
		console.error(`Available providers: ${validProviders.join(", ")}`);
		process.exit(1);
	}

	// Skip Anthropic since we're bridging to non-Anthropic providers
	if (provider === "anthropic") {
		console.error(`❌ Anthropic provider not supported for bridging`);
		const validProviders = getNonAnthropicProviders();
		console.error(`Available providers: ${validProviders.join(", ")}`);
		process.exit(1);
	}

	const models = getCapableModelsForProvider(provider);

	if (models.length === 0) {
		console.error(`❌ No capable models found for provider: ${provider}`);
		const validProviders = getNonAnthropicProviders();
		console.error(`Available providers: ${validProviders.join(", ")}`);
		process.exit(1);
	}

	// Get provider display name with exhaustive switch (provider is already validated as non-anthropic)
	let providerDisplayName: string;
	if (provider === "openai") {
		providerDisplayName = "OpenAI";
	} else if (provider === "google") {
		providerDisplayName = "Google";
	} else {
		// This should never happen since we validated provider above
		console.error(`❌ Unexpected provider: ${provider}`);
		process.exit(1);
	}

	console.log(`${providerDisplayName} models with tools and image support:\n`);
	console.log(
		`  ${"Model".padEnd(35)} ${"Tools".padStart(6)}  ${"Images".padStart(7)}  ${"Max Input".padStart(12)}  ${"Max Output".padStart(12)}`,
	);
	console.log(
		`  ${"".padEnd(35, "─")} ${"".padStart(6, "─")}  ${"".padStart(7, "─")}  ${"".padStart(12, "─")}  ${"".padStart(12, "─")}`,
	);

	// Sort models by max input tokens (descending) to show most capable first
	const sortedModels = models
		.map((model) => ({ model, data: findModelData(model) }))
		.filter((item) => item.data !== undefined)
		.sort((a, b) => b.data!.contextWindow - a.data!.contextWindow)
		.map((item) => item.model);

	for (const model of sortedModels) {
		const data = findModelData(model);
		if (data) {
			console.log(formatModelInfo(model, data));
		}
	}

	console.log(`\nUsage:`);
	console.log(`  claude-bridge ${provider} <model>     Run with specific model`);
	console.log(`  claude-bridge --help                  Show detailed help`);
	console.log(`\nExamples:`);
	console.log(`  claude-bridge ${provider} ${sortedModels[0]}`);
	if (sortedModels[1]) {
		console.log(`  claude-bridge ${provider} ${sortedModels[1]}`);
	}
}

function parseArguments(argv: string[]): ParsedArgs {
	const args: ParsedArgs = {
		toolsEnabled: true, // デフォルトでツールを有効化
		claudeArgs: [],
	};

	let i = 2; // Skip 'node' and script name

	while (i < argv.length) {
		const arg = argv[i];

		if (arg === "--version") {
			args.version = true;
			i++;
		} else if (arg === "--help" || arg === "-h") {
			args.help = true;
			i++;
		} else if (arg === "--trace") {
			args.trace = true;
			i++;
		} else if (arg === "--apiKey") {
			if (i + 1 < argv.length && argv[i + 1] !== undefined) {
				args.apiKey = argv[++i];
			}
			i++;
		} else if (arg === "--baseURL") {
			if (i + 1 < argv.length && argv[i + 1] !== undefined) {
				args.baseURL = argv[++i];
			}
			i++;
		} else if (arg === "--maxRetries") {
			if (i + 1 < argv.length && argv[i + 1] !== undefined) {
				const nextArg = argv[++i];
				if (nextArg !== undefined) {
					const retries = parseInt(nextArg, 10);
					if (isNaN(retries) || retries < 0) {
						console.error(`❌ Invalid --maxRetries value: ${nextArg}`);
						process.exit(1);
					}
					args.maxRetries = retries;
				}
			}
			i++;
		} else if (arg === "--log-dir") {
			if (i + 1 < argv.length && argv[i + 1] !== undefined) {
				args.logDir = argv[++i];
			}
			i++;
		} else if (arg === "--patch-claude") {
			args.patchClaude = true;
			i++;
		} else if (arg === "--tools-enable") {
			args.toolsEnabled = true;
			i++;
		} else if (arg === "--tools-disable") {
			args.toolsEnabled = false;
			i++;
		} else if (arg === "--debug") {
			args.debug = true;
			i++;
		} else if (arg && arg.startsWith("--")) {
			// Unknown option, add to Claude args
			args.claudeArgs.push(arg);
			// Check if this option takes a value
			if (i + 1 < argv.length) {
				const nextArg = argv[i + 1];
				if (nextArg !== undefined && !nextArg.startsWith("--")) {
					args.claudeArgs.push(nextArg);
					i++;
				}
			}
			i++;
		} else if (!args.provider && arg && !arg.startsWith("-")) {
			// First non-option argument is provider
			args.provider = arg;
			i++;
		} else if (!args.model && arg && !arg.startsWith("-")) {
			// Second non-option argument is model
			args.model = arg;
			i++;
		} else {
			// Everything else goes to Claude args
			if (arg !== undefined) {
				args.claudeArgs.push(arg);
			}
			i++;
		}
	}

	return args;
}

function validateProviderAndModel(provider?: string, model?: string): { provider: Provider; model: string } | null {
	if (!provider) return null;

	// Validate provider with type checking
	if (!isValidProvider(provider)) {
		console.error(`❌ Invalid provider: ${provider}`);
		const validProviders = getNonAnthropicProviders();
		console.error(`Available providers: ${validProviders.join(", ")}`);
		return null;
	}

	// Skip Anthropic since we're bridging to non-Anthropic providers
	if (provider === "anthropic") {
		console.error(`❌ Anthropic provider not supported for bridging`);
		const validProviders = getNonAnthropicProviders();
		console.error(`Available providers: ${validProviders.join(", ")}`);
		return null;
	}

	if (!model) return null;

	// Get capable models for this provider
	const capableModels = getCapableModelsForProvider(provider);

	// Check if model is in our known capable models
	if (!capableModels.includes(model)) {
		console.warn(`⚠️  Unknown model for ${provider}: ${model}`);
		console.warn(`   This model is not in our registry but will be attempted.`);
		console.warn(`   Known ${provider} models:`);
		for (const availableModel of capableModels.slice(0, 3)) {
			// Show first 3
			console.warn(`     ${availableModel}`);
		}
		if (capableModels.length > 3) {
			console.warn(`     ... and ${capableModels.length - 3} more`);
		}
		console.warn(`   Run 'claude-bridge ${provider}' to see all known models\n`);
	}

	// Check if model data exists in our registry (for capability info)
	const modelData = findModelData(model);
	if (!modelData) {
		console.warn(`⚠️  Model capabilities unknown for: ${model}`);
		console.warn(`   Tool and image support cannot be validated.\n`);
	}

	return { provider, model };
}

function findClaudeExecutable(): string | null {
	// Try to find claude in PATH
	const possibleNames = ["claude", "claude-code"];

	for (const name of possibleNames) {
		try {
			const result = spawnSync("which", [name], { stdio: "pipe", encoding: "utf-8" });
			if (result.status === 0) {
				return result.stdout.trim();
			}
		} catch {
			// Continue searching
		}
	}

	// If not found, return null
	return null;
}

function runClaudeWithBridge(args: ClaudeArgs): number {
	// Validate we have required provider and model
	if (!args.provider || !args.model) {
		console.error("❌ Internal error: provider and model are required");
		return 1;
	}

	// Get API key with exhaustive switch
	let apiKey = args.apiKey;
	if (!apiKey) {
		let envVar: string;
		switch (args.provider) {
			case "anthropic":
				envVar = "ANTHROPIC_API_KEY";
				break;
			case "openai":
				envVar = "OPENAI_API_KEY";
				break;
			case "google":
				envVar = "GOOGLE_API_KEY";
				break;
			default:
				// TypeScript will catch if we miss any provider cases
				const _exhaustiveCheck: never = args.provider;
				return _exhaustiveCheck;
		}

		apiKey = process.env[envVar];
		if (!apiKey) {
			console.error(`❌ API key not found. Provide --apiKey or set ${envVar} environment variable`);
			return 1;
		}
	}

	// Check model capabilities and warn if Claude requests exceed limits
	const modelData = findModelData(args.model);
	if (modelData) {
		console.log(`🌉 Claude Bridge starting:`);
		console.log(`   Provider: ${args.provider}`);
		console.log(`   Model: ${args.model}`);
		console.log(`   Max Input Tokens: ${modelData.contextWindow.toLocaleString()}`);
		console.log(`   Max Output Tokens: ${modelData.maxOutputTokens.toLocaleString()}`);
		console.log(`   Tools Support: ${modelData.supportsTools ? "✓" : "✗"}`);
		console.log(`   Images Support: ${modelData.supportsImageInput ? "✓" : "✗"}`);
		if (args.debug) {
			console.log(`   Logging to: ${args.logDir || ".claude-bridge"}/requests.jsonl`);
		}
	} else {
		console.log(`🌉 Claude Bridge starting:`);
		console.log(`   Provider: ${args.provider}`);
		console.log(`   Model: ${args.model}`);
		if (args.debug) {
			console.log(`   Logging to: ${args.logDir || ".claude-bridge"}/requests.jsonl`);
		}
	}

	let claudeExe = findClaudeExecutable();
	if (!claudeExe) {
		console.error("❌ Claude CLI not found in PATH");
		console.error("❌ Please install Claude Code CLI first");
		return 1;
	}

	// Patch Claude binary if requested
	if (args.patchClaude) {
		console.log("🔧 Patching Claude binary to disable anti-debugging...");
		const logDir = args.logDir || ".claude-bridge";
		claudeExe = patchClaudeBinary(claudeExe, logDir);
	}

	const __dirname = path.dirname(fileURLToPath(import.meta.url));
	const interceptorLoader = path.join(__dirname, "interceptor-loader.js");

	// Filter out debugging flags from node arguments
	const cleanNodeArgs = ["--import", interceptorLoader];

	const spawnArgs = [...cleanNodeArgs, claudeExe, ...args.claudeArgs];

	console.log(`🚀 Launching: node ${spawnArgs.join(" ")}`);

	// Clean environment to avoid Claude's anti-debugging checks
	const cleanEnv = { ...process.env };

	const result = spawnSync("node", spawnArgs, {
		stdio: "inherit",
		env: {
			...cleanEnv,
			NODE_OPTIONS: `${cleanEnv["NODE_OPTIONS"]} --no-deprecation`,
			CLAUDE_BRIDGE_PROVIDER: args.provider,
			CLAUDE_BRIDGE_MODEL: args.model,
			CLAUDE_BRIDGE_API_KEY: apiKey,
			CLAUDE_BRIDGE_BASE_URL: args.baseURL,
			CLAUDE_BRIDGE_MAX_RETRIES: args.maxRetries?.toString(),
			CLAUDE_BRIDGE_LOG_DIR: args.logDir,
			CLAUDE_BRIDGE_DEBUG: args.debug?.toString(),
			CLAUDE_BRIDGE_TRACE: args.trace?.toString(),
			CLAUDE_BRIDGE_TOOLS_ENABLED: (args.toolsEnabled ?? true).toString(),
		},
	});

	if (result.error) {
		console.error(`❌ Failed to start Claude: ${result.error.message}`);
		return 1;
	}

	if (result.signal) {
		console.log(`\n🛑 Claude terminated by signal: ${result.signal}`);
	} else {
		console.log(`\n✅ Claude exited with code: ${result.status}`);
	}

	return result.status || 0;
}

async function main(argv: string[] = process.argv) {
	const parsedArgs = parseArguments(argv);

	// Handle version
	if (parsedArgs.version) {
		console.log(VERSION);
		return;
	}

	// Handle help
	if (parsedArgs.help) {
		showHelp();
		return;
	}

	// Handle trace mode - log requests and call original Anthropic API
	if (parsedArgs.trace) {
		const exitCode = runClaudeWithBridge({
			provider: "anthropic", // dummy, ignored in trace mode
			model: "claude-3-5-sonnet-20241022", // dummy, ignored in trace mode
			trace: true,
			debug: true, // trace implies debug
			claudeArgs: parsedArgs.claudeArgs,
			// All other flags ignored in trace mode
		});
		process.exit(exitCode);
	}

	// Handle no arguments - show providers
	if (!parsedArgs.provider) {
		showProviders();
		return;
	}

	// Handle provider only - show models for that provider
	if (parsedArgs.provider && !parsedArgs.model) {
		showProviderModels(parsedArgs.provider);
		return;
	}

	// Handle provider + model - validate and execute
	const validated = validateProviderAndModel(parsedArgs.provider, parsedArgs.model);
	if (!validated) {
		process.exit(1);
	}

	const exitCode = runClaudeWithBridge({
		provider: validated.provider,
		model: validated.model,
		apiKey: parsedArgs.apiKey,
		baseURL: parsedArgs.baseURL,
		maxRetries: parsedArgs.maxRetries,
		logDir: parsedArgs.logDir,
		patchClaude: parsedArgs.patchClaude || false,
		debug: parsedArgs.debug || false,
		trace: parsedArgs.trace || false,
		toolsEnabled: parsedArgs.toolsEnabled,
		claudeArgs: parsedArgs.claudeArgs,
	});

	process.exit(exitCode);
}

process.on("unhandledRejection", (reason, promise) => {
	console.error("Unhandled Rejection at:", promise, "reason:", reason);
	process.exit(1);
});

// Export functions for testing
export default main;
export { runClaudeWithBridge, parseArguments, validateProviderAndModel, getCapableModels };

// Only run if this file is executed directly (ESM check)
// Handle both direct execution and symlink execution
const isMainModule =
	import.meta.url === `file://${process.argv[1]}` ||
	import.meta.url === `file://${process.argv[1]?.replace(/\.js$/, ".mjs")}` ||
	process.argv[1]?.endsWith("claude-bridge");

if (isMainModule) {
	main().catch((error) => {
		console.error("Fatal error:", error);
		process.exit(1);
	});
}
