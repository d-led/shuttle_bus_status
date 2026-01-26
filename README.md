# shuttle_bus_status

Mono-repo containing:

- `camera/`: camera sampling + plate detection pipeline (Raspberry Pi / macOS dev)
- `server/`: PyView-based UI (camera “live” sampled snapshots)

## Setup

Use the unified setup script from the repo root:

```bash
scripts/setup.sh
```

## Run camera server

```bash
scripts/start_camera_server.sh
```

