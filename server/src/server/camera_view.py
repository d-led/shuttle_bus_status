"""Camera server LiveView.

This module uses `pyview`'s `PyView` Starlette integration. LiveViews are not ASGI apps
themselves; they must be registered via `PyView.add_live_view(...)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyview import LiveView
from pyview.live_view import LiveRender, LiveTemplate
from pyview.template import Template

if TYPE_CHECKING:
    from pyview.meta import PyViewMeta

_CAMERA_VIEW_TEMPLATE = LiveTemplate(
    Template(
        """
<div class="section">
  <h2>
    <span class="{{ camera_status_class }}"></span>
    System Status
  </h2>
  <div class="info-grid">
    <div class="info-card">
      <div class="label">Status</div>
      <div class="value">{{ status_text }}</div>
    </div>
    <div class="info-card">
      <div class="label">Camera</div>
      <div class="value">{{ camera_status }}</div>
    </div>
    <div class="info-card">
      <div class="label">Plates Detected</div>
      <div class="value">{{ plates_detected }}</div>
    </div>
  </div>
</div>

<div class="section">
  <h2>Camera Feed</h2>
  <div class="placeholder">
    <div class="placeholder-icon">📹</div>
    <p>Camera feed will appear here</p>
    <p class="placeholder-sub">Video streaming will be implemented soon</p>
  </div>
</div>

<div class="section">
  <h2>Recent Activity</h2>
  <div class="placeholder">
    <div class="placeholder-icon">📋</div>
    <p>Plate detection events will appear here</p>
    <p class="placeholder-sub">Logs and events will be displayed in real-time</p>
  </div>
</div>
""".strip(),
        template_id="server.camera_view",
    )
)


class CameraLiveView(LiveView[dict[str, Any]]):
    """Camera monitoring view.

    Works even when no camera is connected; the UI will just show "Not Connected".
    """

    async def mount(self, socket: Any, session: Any) -> None:
        del session
        socket.context = {
            "title": "Shuttle Bus Status",
            "status": "ready",
            "camera_connected": False,
            "plates_detected": 0,
        }
        socket.live_title = "Shuttle Bus Status"

    async def render(self, assigns: dict[str, Any], meta: PyViewMeta) -> LiveRender:
        camera_connected = bool(assigns.get("camera_connected"))
        status = str(assigns.get("status", "ready"))

        computed = {
            **assigns,
            "status_text": "Ready" if status == "ready" else "Initializing...",
            "camera_status": "Connected" if camera_connected else "Not Connected",
            "camera_status_class": (
                "status-indicator connected" if camera_connected else "status-indicator"
            ),
            "plates_detected": int(assigns.get("plates_detected", 0)),
        }

        return LiveRender(_CAMERA_VIEW_TEMPLATE, computed, meta)
