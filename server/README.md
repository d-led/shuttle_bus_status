# Shuttle Bus Status Server

Server component for shuttle bus status monitoring using pyview-web for live updates.

## Quick Start

1. Run the setup script:
   
   **For development (includes dev tools):**
   ```bash
   ./scripts/setup.sh
   ```
   
   **For production (Raspberry Pi - no dev tools):**
   ```bash
   ./scripts/setup_prod.sh
   ```
   
   This works on macOS, Linux, and Raspberry Pi.

2. Start the camera server:
   ```bash
   ./scripts/start_camera_server.sh
   ```
   The server will be available at `http://localhost:8000` (or the configured port).

3. Alternatively, activate the virtual environment and run manually:
   ```bash
   source .venv/bin/activate
   python -m server.main
   # or
   shuttle-bus-status-server
   ```

## Platform Support

- **macOS**: Uses `.venv` in project root. Full support for development and testing.
- **Linux**: Uses `.venv` in project root. Works on Ubuntu, Debian, and other distributions.
- **Raspberry Pi**: Uses `.venv` in project root (unified location). Use `setup_prod.sh` for production deployment.

## Configuration

Server configuration is managed via `config.toml` in the project root:

```toml
[server]
host = "0.0.0.0"
port = 8000
debug = false

[public_server]
host = "0.0.0.0"
port = 8000
debug = false
```

- `[server]` - Configuration for the main server
- `[public_server]` - Configuration for the camera server (Raspberry Pi or Mac)

## Development

Run tests:

```bash
./scripts/test.sh
```
