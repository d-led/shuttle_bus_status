"""Tests for server configuration management."""

from pathlib import Path

import pytest

from server.config import ServerSettings, Settings


def test_default_server_settings() -> None:
    """Test that default server settings are applied correctly."""
    settings = ServerSettings()

    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.debug is False


def test_custom_server_settings() -> None:
    """Test creating server settings with custom values."""
    settings = ServerSettings(host="127.0.0.1", port=9000, debug=True)

    assert settings.host == "127.0.0.1"
    assert settings.port == 9000
    assert settings.debug is True


def test_port_validation() -> None:
    """Test that port validation works correctly."""
    # Valid ports
    ServerSettings(port=1)
    ServerSettings(port=65535)
    ServerSettings(port=8000)

    # Invalid ports
    with pytest.raises(Exception):  # pydantic validation error
        ServerSettings(port=0)

    with pytest.raises(Exception):  # pydantic validation error
        ServerSettings(port=65536)


def test_default_settings() -> None:
    """Test that default settings include both server configs."""
    settings = Settings()

    assert settings.server.host == "0.0.0.0"
    assert settings.server.port == 8000
    assert settings.server.debug is False

    assert settings.public_server.host == "0.0.0.0"
    assert settings.public_server.port == 8000
    assert settings.public_server.debug is False


def test_load_config_from_project_root(tmp_path: Path) -> None:
    """Test loading configuration from project root."""
    # Create a config.toml file
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[server]
host = "127.0.0.1"
port = 9000
debug = true

[public_server]
host = "0.0.0.0"
port = 8080
debug = false
""")

    # Change to the temp directory
    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        settings = Settings.load_from_project_root()

        assert settings.server.host == "127.0.0.1"
        assert settings.server.port == 9000
        assert settings.server.debug is True

        assert settings.public_server.host == "0.0.0.0"
        assert settings.public_server.port == 8080
        assert settings.public_server.debug is False
    finally:
        os.chdir(original_cwd)


def test_load_config_with_partial_server_config(tmp_path: Path) -> None:
    """Test loading config when only server section is provided."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[server]
port = 3000
""")

    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        settings = Settings.load_from_project_root()

        # Server should use provided port, defaults for rest
        assert settings.server.port == 3000
        assert settings.server.host == "0.0.0.0"  # default
        assert settings.server.debug is False  # default

        # Public server should use all defaults
        assert settings.public_server.port == 8000  # default
        assert settings.public_server.host == "0.0.0.0"  # default
        assert settings.public_server.debug is False  # default
    finally:
        os.chdir(original_cwd)


def test_load_config_with_partial_public_server_config(tmp_path: Path) -> None:
    """Test loading config when only public_server section is provided."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[public_server]
port = 5000
debug = true
""")

    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        settings = Settings.load_from_project_root()

        # Public server should use provided values
        assert settings.public_server.port == 5000
        assert settings.public_server.debug is True
        assert settings.public_server.host == "0.0.0.0"  # default

        # Server should use all defaults
        assert settings.server.port == 8000  # default
        assert settings.server.host == "0.0.0.0"  # default
        assert settings.server.debug is False  # default
    finally:
        os.chdir(original_cwd)


def test_load_config_finds_project_root(tmp_path: Path) -> None:
    """Test that config loading finds project root in parent directories."""
    # Create config.toml in tmp_path
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[server]
port = 7000
""")

    # Create a subdirectory
    subdir = tmp_path / "subdir" / "nested"
    subdir.mkdir(parents=True)

    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(subdir)
        settings = Settings.load_from_project_root(project_root=tmp_path)

        assert settings.server.port == 7000
    finally:
        os.chdir(original_cwd)


def test_load_config_missing_file_raises_error(tmp_path: Path) -> None:
    """Test that missing config file raises appropriate error."""
    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        # No config.toml file exists

        with pytest.raises(FileNotFoundError) as exc_info:
            Settings.load_from_project_root()

        assert "config.toml" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()
    finally:
        os.chdir(original_cwd)


def test_load_config_with_explicit_project_root(tmp_path: Path) -> None:
    """Test loading config with explicitly provided project root."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[server]
port = 6000
""")

    # Change to a different directory
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(other_dir)
        settings = Settings.load_from_project_root(project_root=tmp_path)

        assert settings.server.port == 6000
    finally:
        os.chdir(original_cwd)
