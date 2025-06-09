#!/bin/bash

# test-env.sh - Script to test environment variable handling in claude script

echo "Testing claude script with environment variables..."
echo ""

# Export test environment variables
export CLAUDE_BRIDGE_PROVIDER="openai"
export CLAUDE_BRIDGE_MODEL="gpt-4o"
export CLAUDE_BRIDGE_API_KEY="sk-testkey123456789"
export CLAUDE_BRIDGE_BASE_URL="https://test-api.example.com/v1"

# Run claude with -h flag to check if it picks up the environment variables
# This should display the help message without actually running the model
echo "Running 'claude -h' to check environment variable handling:"
claude -h

echo ""
echo "Check /tmp/claude-bridge-intercept.log for detailed logging"
echo "Relevant log line should show:"
echo "Environment: PROVIDER=openai, MODEL=gpt-4o, BASE_URL=https://test-api.example.com/v1, API_KEY=sk-te..."
echo ""

# Print the most recent log entry
echo "Last log entries:"
tail -n 10 /tmp/claude-bridge-intercept.log