"""Server configuration management."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]  # Python < 3.11


class ServerSettings(BaseSettings):
    """Server configuration."""

    host: str = Field(
        default="0.0.0.0",
        description="Server host address",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port",
    )
    debug: bool = Field(
        default=False,
        description="Whether to enable debug mode",
    )


class Settings(BaseSettings):
    """Main server settings."""

    model_config = SettingsConfigDict(
        env_file=None,  # We use TOML, not .env
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def load_from_project_root(cls, project_root: Path | None = None) -> "Settings":
        """Load settings from config.toml in project root."""
        if project_root is None:
            # Try to find project root by looking for config.toml
            current = Path.cwd()
            for parent in [current, *current.parents]:
                if (parent / "config.toml").exists():
                    project_root = parent
                    break
            if project_root is None:
                project_root = current

        config_file = project_root / "config.toml"
        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}. "
                "Please create config.toml in the project root."
            )

        with config_file.open("rb") as f:
            toml_data = tomllib.load(f)

        return cls(**toml_data)
