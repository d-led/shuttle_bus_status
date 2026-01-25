"""Tests for server root template robustness."""

from server.root_template import camera_root_template


def test_camera_root_template_renders_with_minimal_context() -> None:
    template = camera_root_template(
        theme="light",
        css_vars={"banner_color": "#000"},
    )

    html = template(
        {
            "id": "123",
            "content": "<div>Hello</div>",
            "title": "Title",
            "csrf_token": "csrf",
            "session": "{}",
            "additional_head_elements": [],
        }
    )

    assert "/static/assets/app.js" in html
    assert "daisyui" in html
    # Critical: the live view container must NOT be HTML-escaped.
    assert 'data-phx-main="true"' in html
    assert "&lt;div" not in html
