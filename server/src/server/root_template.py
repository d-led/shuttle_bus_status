"""PyView root templates.

We keep a small Python wrapper (as PyView expects `RootTemplate = Callable[[ctx], str]`)
but the actual HTML is in `root_template.html`, loaded like MVG/Monty templates
(ibis + FileReloader).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from markupsafe import Markup
from pyview.live_socket import UnconnectedSocket
from pyview.meta import PyViewMeta
from pyview.template.live_template import LiveTemplate
from pyview.vendor import ibis
from pyview.vendor.ibis.loaders import FileReloader

if TYPE_CHECKING:
    from pyview.template.root_template import (
        ContentWrapper,
        RootTemplate,
        RootTemplateContext,
    )


def _load_root_template() -> LiveTemplate:
    current_file_path = Path(__file__).resolve()
    templates_dir = current_file_path.parent

    if not hasattr(ibis, "loader") or not isinstance(ibis.loader, FileReloader):
        ibis.loader = FileReloader(str(templates_dir))

    template_file = templates_dir / "root_template.html"
    template_content = template_file.read_text(encoding="utf-8")
    template = ibis.Template(template_content, template_id=str(template_file))
    return LiveTemplate(template)


def camera_root_template(
    *,
    theme: str = "light",
    css_vars: dict[str, str] | None = None,
    content_wrapper: ContentWrapper | None = None,
) -> RootTemplate:
    """Root template styled like MVG/Monty (compact + Tailwind/DaisyUI).

    The user explicitly requested the `<head>` structure (DaisyUI + Tailwind CDN)
    and compactness.
    """

    content_wrapper = content_wrapper or (lambda _ctx, html: html)

    css_var_values = css_vars or {}
    root_template = _load_root_template()

    def template(context: RootTemplateContext) -> str:
        try:
            title = str(context.get("title") or "LiveView")
            # Keep Markup values unescaped (PyView provides <style> tags here).
            additional_head_elements = list(context["additional_head_elements"])

            def _var(key: str, default: str) -> str:
                value = css_var_values.get(key, default)
                return str(value) if value is not None else default

            css_vars_markup = Markup(f"""
    <style>
        :root {{
            --font-size-route-number: {_var("font_size_route_number", "1.15rem")};
            --font-size-destination: {_var("font_size_destination", "1rem")};
            --font-size-platform: {_var("font_size_platform", "0.9rem")};
            --font-size-time: {_var("font_size_time", "1.1rem")};
            --font-size-no-departures: {_var("font_size_no_departures", "1rem")};
            --font-size-direction-header: {_var("font_size_direction_header", "1rem")};
            --font-size-stop-header: {_var("font_size_stop_header", "1rem")};
            --font-size-pagination-indicator: {_var("font_size_pagination_indicator", "0.9rem")};
            --font-size-countdown-text: {_var("font_size_countdown_text", "0.9rem")};
            --font-size-status-header: {_var("font_size_status_header", "0.9rem")};
            --font-size-delay-amount: {_var("font_size_delay_amount", "0.9rem")};
            --banner-color: {_var("banner_color", "#667eea")};
        }}
    </style>
""")

            main_content = content_wrapper(
                context,
                Markup(f"""
      <div
        data-phx-main="true"
        data-phx-session="{context["session"]}"
        data-phx-static=""
        id="phx-{context["id"]}"
        class="min-h-screen bg-base-200 text-[16px] sm:text-[17px] leading-snug"
        >
        <div class="mx-auto max-w-6xl p-3 sm:p-4">
          {context["content"]}
        </div>
      </div>
"""),
            )

            assigns: dict[str, object] = {
                "theme": theme,
                "title": title,
                "csrf_token": str(context["csrf_token"]),
                "additional_head_elements": additional_head_elements,
                "css_vars_markup": css_vars_markup,
                "main_content": main_content,
            }
            return str(
                root_template.render(
                    assigns,
                    meta=PyViewMeta(socket=UnconnectedSocket()),
                )
            )
        except Exception as e:
            # Robustness: never crash the server just because the root template failed.
            # Return a minimal page so polling/background processes (added later) keep running.
            return (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                "<title>LiveView</title></head><body>"
                "<h1>Template error</h1>"
                f"<pre>{e!r}</pre>"
                "</body></html>"
            )

    return template
