"""Main entry point for camera plate detection."""

from camera.config import Settings
from camera.logger import configure_from_settings, get_logger


def main() -> None:
    """Main entry point for camera plate detection."""
    logger = get_logger(__name__)
    logger.info("Starting camera plate detection system")

    # Load configuration
    try:
        settings = Settings.load_from_project_root()
        configure_from_settings(settings)
        logger.info("Configuration loaded", config_file="config.toml")
    except FileNotFoundError as e:
        logger.error("Configuration file not found", error=str(e))
        raise

    logger.info("Camera plate detection system initialized")
    # TODO: Implement main polling loop


if __name__ == "__main__":
    main()
