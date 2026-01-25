"""Pytest configuration for the test suite."""

import sys
from pathlib import Path

# Add server/src to Python path for server tests
# This must happen before any test modules are imported
# conftest.py is automatically loaded by pytest before collecting tests
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SERVER_SRC = PROJECT_ROOT / "server" / "src"
if SERVER_SRC.exists() and SERVER_SRC.is_dir():
    server_src_str = str(SERVER_SRC.resolve())
    # Insert at the beginning to ensure it's checked first
    # Remove it first if it exists to avoid duplicates
    if server_src_str in sys.path:
        sys.path.remove(server_src_str)
    sys.path.insert(0, server_src_str)
    # Verify it works
    try:
        import server.config  # noqa: F401
    except ImportError:
        # If import fails, try one more time after ensuring path is correct
        if server_src_str not in sys.path:
            sys.path.insert(0, server_src_str)
