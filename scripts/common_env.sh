#!/usr/bin/env bash
# Common environment setup script
# Source this in other scripts to get consistent virtual environment and PYTHONPATH setup

# Detect environment FIRST (before using OS_TYPE)
IS_CI="${CI:-false}"
OS_TYPE="unknown"

if [[ "$IS_CI" == "true" ]] || [[ -n "${CI:-}" ]]; then
    OS_TYPE="ci"
elif [[ -f /etc/os-release ]]; then
    if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null || grep -qi "raspberry" /etc/os-release; then
        OS_TYPE="raspberrypi"
    else
        OS_TYPE="linux"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
fi

# Get the project root directory
# Priority: 1) Already set PROJECT_ROOT (from calling script), 2) Current directory with config.toml, 3) Script location
# If PROJECT_ROOT is already set and has config.toml, use it
# Otherwise, find it starting from current directory
if [ -z "${PROJECT_ROOT:-}" ] || [ ! -f "${PROJECT_ROOT}/config.toml" ]; then
    # Start from current directory - this should be the project root when scripts are run
    CURRENT_DIR="$(pwd)"
    
    # Check current directory first (most common case when scripts are run from project root)
    if [ -f "$CURRENT_DIR/config.toml" ]; then
        PROJECT_ROOT="$CURRENT_DIR"
    elif [ -n "${PROJECT_ROOT:-}" ] && [ -f "${PROJECT_ROOT}/config.toml" ]; then
        # PROJECT_ROOT was set and is valid, use it
        :
    else
        # Fallback: calculate from script location (scripts/..)
        COMMON_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        CALCULATED_ROOT="$(cd "$COMMON_ENV_SCRIPT_DIR/.." && pwd)"
        # Verify this has config.toml
        if [ -f "$CALCULATED_ROOT/config.toml" ]; then
            PROJECT_ROOT="$CALCULATED_ROOT"
        else
            # Use calculated root anyway (better than nothing)
            PROJECT_ROOT="$CALCULATED_ROOT"
        fi
    fi
    export PROJECT_ROOT
fi

# Determine virtual environment location
if [[ "$OS_TYPE" == "raspberrypi" ]]; then
    # Raspberry Pi: prefer server-venv, fallback to camera-venv
    if [ -d "$HOME/server-venv" ]; then
        VENV_DIR="$HOME/server-venv"
    elif [ -d "$HOME/camera-venv" ]; then
        VENV_DIR="$HOME/camera-venv"
    else
        VENV_DIR="$HOME/camera-venv"  # Default for RPi
    fi
else
    # macOS, Linux, CI: use .venv in project root
    VENV_DIR="$PROJECT_ROOT/.venv"
    # Also check for camera/.venv as fallback
    if [ ! -d "$VENV_DIR" ] && [ -d "$PROJECT_ROOT/camera/.venv" ]; then
        VENV_DIR="$PROJECT_ROOT/camera/.venv"
    fi
fi

# Activate virtual environment if it exists
# Check multiple possible locations
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    export VIRTUAL_ENV_ACTIVATED=true
elif [ -d "$PROJECT_ROOT/.venv" ] && [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # Fallback: try project root .venv directly
    VENV_DIR="$PROJECT_ROOT/.venv"
    source "$VENV_DIR/bin/activate"
    export VIRTUAL_ENV_ACTIVATED=true
elif [ -d "$PROJECT_ROOT/camera/.venv" ] && [ -f "$PROJECT_ROOT/camera/.venv/bin/activate" ]; then
    # Fallback: try camera/.venv
    VENV_DIR="$PROJECT_ROOT/camera/.venv"
    source "$VENV_DIR/bin/activate"
    export VIRTUAL_ENV_ACTIVATED=true
else
    export VIRTUAL_ENV_ACTIVATED=false
fi

# Set PYTHONPATH for server tests (needs to be available throughout)
export PYTHONPATH="${PROJECT_ROOT}/server/src:${PYTHONPATH:-}"

# Export environment variables for use in other scripts
export OS_TYPE
export IS_CI
export VENV_DIR
