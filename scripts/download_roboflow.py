#!/usr/bin/env python3
"""Download German License Plates dataset from Roboflow."""

import os
import sys

try:
    from roboflow import Roboflow

    # Roboflow requires an API key even for public datasets
    # Check for API key in environment variable or config
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")

    if not api_key:
        # Try to read from config file if it exists
        config_path = os.path.expanduser("~/.roboflow/config")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    for line in f:
                        if line.startswith("api_key"):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
            except Exception:
                pass

    if not api_key:
        print("NO_API_KEY")
        print("Roboflow requires an API key even for public datasets.")
        print("Get your API key from: https://app.roboflow.com/")
        print("Then set: export ROBOFLOW_API_KEY='your-api-key'")
        sys.exit(1)

    # Initialize Roboflow with API key
    rf = Roboflow(api_key=api_key)

    # Try to access the public dataset
    try:
        project = rf.workspace("max-mustermann-gmm7j").project("german-license-plates-hptbz")
        # Get latest version (version 7 has the most images: 1243)
        # Try to get the latest version number, or just download latest
        try:
            versions = project.list_versions()
            if versions and len(versions) > 0:
                # Get the version with the most images (usually the latest)
                latest_version = max(versions, key=lambda v: v.get("images", 0))
                version_num = latest_version["id"].split("/")[-1]
                dataset = project.version(int(version_num)).download(
                    "yolov8", location=sys.argv[1] if len(sys.argv) > 1 else "."
                )
            else:
                # Fallback: try version 7 (latest based on test)
                dataset = project.version(7).download(
                    "yolov8", location=sys.argv[1] if len(sys.argv) > 1 else "."
                )
        except Exception as ve:
            # If version detection fails, try downloading latest without version number
            try:
                dataset = project.download(
                    "yolov8", location=sys.argv[1] if len(sys.argv) > 1 else "."
                )
            except Exception:
                # Last resort: try version 7
                dataset = project.version(7).download(
                    "yolov8", location=sys.argv[1] if len(sys.argv) > 1 else "."
                )
        print("SUCCESS")
    except Exception as e:
        print(f"API_ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
except ImportError:
    print("ROBOFLOW_NOT_INSTALLED")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
