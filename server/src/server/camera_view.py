"""Camera server LiveView for Raspberry Pi."""

from typing import Any

from pyview import LiveView


def app_frame(title: str, content: str) -> str:
    """Application frame with consistent styling."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            color: #333;
        }}

        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .header h1 {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #2d3748;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .header .icon {{
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}

        .main {{
            flex: 1;
            padding: 2rem;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .container {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            max-width: 1200px;
            width: 100%;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #48bb78;
            margin-right: 0.5rem;
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: 0.5;
            }}
        }}

        .section {{
            margin-bottom: 2rem;
        }}

        .section:last-child {{
            margin-bottom: 0;
        }}

        .section h2 {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}

        .info-card {{
            background: #f7fafc;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }}

        .info-card .label {{
            font-size: 0.875rem;
            color: #718096;
            margin-bottom: 0.25rem;
        }}

        .info-card .value {{
            font-size: 1.125rem;
            font-weight: 600;
            color: #2d3748;
        }}

        .placeholder {{
            text-align: center;
            padding: 3rem 2rem;
            color: #718096;
        }}

        .placeholder-icon {{
            font-size: 4rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }}

        @media (max-width: 768px) {{
            .header {{
                padding: 1rem;
            }}

            .header h1 {{
                font-size: 1.25rem;
            }}

            .main {{
                padding: 1rem;
            }}

            .container {{
                padding: 1.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>
            <span class="icon">🚌</span>
            {title}
        </h1>
    </div>
    <div class="main">
        <div class="container">
            {content}
        </div>
    </div>
</body>
</html>
"""


class CameraLiveView(LiveView):
    """Camera monitoring LiveView for Raspberry Pi."""

    async def mount(self, _session: Any, _params: Any) -> dict[str, Any]:
        """Initialize the live view state."""
        return {
            "title": "Shuttle Bus Status",
            "status": "ready",
            "camera_connected": False,
            "plates_detected": 0,
            "last_update": None,
        }

    async def render(self, state: dict[str, Any]) -> str:
        """Render the camera monitoring page."""
        status_text = "Ready" if state.get("status") == "ready" else "Initializing..."
        camera_status = (
            "Connected" if state.get("camera_connected") else "Not Connected"
        )
        camera_status_class = (
            "status-indicator" if state.get("camera_connected") else ""
        )

        content = f"""
        <div class="section">
            <h2>
                <span class="{camera_status_class}"></span>
                System Status
            </h2>
            <div class="info-grid">
                <div class="info-card">
                    <div class="label">Status</div>
                    <div class="value">{status_text}</div>
                </div>
                <div class="info-card">
                    <div class="label">Camera</div>
                    <div class="value">{camera_status}</div>
                </div>
                <div class="info-card">
                    <div class="label">Plates Detected</div>
                    <div class="value">{state.get("plates_detected", 0)}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Camera Feed</h2>
            <div class="placeholder">
                <div class="placeholder-icon">📹</div>
                <p>Camera feed will appear here</p>
                <p style="font-size: 0.875rem; margin-top: 0.5rem; opacity: 0.7;">
                    Video streaming will be implemented soon
                </p>
            </div>
        </div>

        <div class="section">
            <h2>Recent Activity</h2>
            <div class="placeholder">
                <div class="placeholder-icon">📋</div>
                <p>Plate detection events will appear here</p>
                <p style="font-size: 0.875rem; margin-top: 0.5rem; opacity: 0.7;">
                    Logs and events will be displayed in real-time
                </p>
            </div>
        </div>
        """

        return app_frame(state.get("title", "Shuttle Bus Status"), content)
