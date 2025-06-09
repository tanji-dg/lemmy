#!/bin/bash

# launch-claude-bridge.sh - Script to launch claude-bridge with proper Node environment
# This script ensures the correct NVM environment is used when launching claude-bridge from VSCode

# Source NVM
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"

# Check if NVM is available
if [ -z "$(command -v nvm)" ]; then
    echo "Error: NVM is not available. Please make sure NVM is installed and properly configured."
    exit 1
fi

# Get the path to the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Use the version of Node required by the project
NODE_VERSION=$(node -e "console.log(require('$SCRIPT_DIR/package.json').engines.node.replace('>=', ''))")
echo "Using Node version $NODE_VERSION from package.json engines field"
nvm use "$NODE_VERSION" || nvm use default

# Check for arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <provider> [model] [options]"
    echo "Example: $0 openai gpt-4o"
    exit 1
fi

# Run claude-bridge with provided arguments
echo "Launching claude-bridge with Node $(node --version)"
npx --no-install claude-bridge "$@"

# Exit with the same status as claude-bridge
exit $?