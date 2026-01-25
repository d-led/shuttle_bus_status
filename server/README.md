# Shuttle Bus Status Server

Server component for shuttle bus status monitoring using pyview-web for live updates.

## Setup

Install dependencies:

```bash
cd server
pip install -e ".[dev]"
```

## Running

From the project root:

```bash
python -m server.main
```

Or use the entry point:

```bash
shuttle-bus-status-server
```

## Configuration

Server configuration is managed via `config.toml` in the project root:

```toml
[server]
host = "0.0.0.0"
port = 8000
debug = false
```

## Development

Run tests:

```bash
./scripts/test.sh
```
