#!/usr/bin/env python3
"""Download German License Plates dataset from Roboflow.

To download manually:
1. Visit: https://universe.roboflow.com/max-mustermann-gmm7j/german-license-plates-hptbz
2. Click on a version (e.g., "Version 7" or latest version)
3. Look for "Download" or "Export" button in the version page
4. Select format: YOLOv8
5. Download and extract to the target directory
"""

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
        print("")
        print("Get your WORKSPACE API key:")
        print("  1. Go to: https://app.roboflow.com/")
        print("  2. Click on your workspace name (top left)")
        print("  3. Go to 'Workspace Settings' or 'Account Settings'")
        print("  4. Find 'Roboflow API' section")
        print("  5. Copy your WORKSPACE API key (not project-specific key)")
        print("  6. Set: export ROBOFLOW_API_KEY='your-workspace-api-key'")
        print("")
        print("Alternative: Download manually from the web interface:")
        print("  1. Visit: https://universe.roboflow.com/max-mustermann-gmm7j/german-license-plates-hptbz")
        print("  2. Click on a version number (e.g., 'Version 7')")
        print("  3. Click 'Download Dataset' button (top right)")
        print("  4. Select format: YOLOv8")
        print("  5. Download ZIP and extract to target directory")
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
                output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
                dataset = project.version(int(version_num)).download(
                    "yolov8", location=output_dir
                )
            else:
                # Fallback: try version 7 (latest based on test)
                output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
                dataset = project.version(7).download(
                    "yolov8", location=output_dir
                )
        except Exception as ve:
            # If version detection fails, try downloading latest without version number
            try:
                output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
                dataset = project.download("yolov8", location=output_dir)
            except Exception:
                # Last resort: try version 7
                output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
                dataset = project.version(7).download("yolov8", location=output_dir)

        # Verify download actually happened
        # Roboflow downloads to subdirectories: train/images/, test/images/, valid/images/
        # The download is async, so we need to wait for it to complete
        import glob
        import time

        output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
        
        # Wait for download/extraction to complete (Roboflow shows progress bars)
        # Give it up to 30 seconds
        max_wait = 30
        wait_interval = 2
        elapsed = 0
        
        while elapsed < max_wait:
            # Search recursively for images (Roboflow creates train/, test/, valid/ subdirs)
            images = (
                glob.glob(os.path.join(output_dir, "**", "*.jpg"), recursive=True)
                + glob.glob(os.path.join(output_dir, "**", "*.png"), recursive=True)
                + glob.glob(os.path.join(output_dir, "**", "*.jpeg"), recursive=True)
            )
            
            if len(images) > 0:
                print(f"SUCCESS: {len(images)} images downloaded")
                break
            
            time.sleep(wait_interval)
            elapsed += wait_interval
        else:
            # Final check after max wait
            images = (
                glob.glob(os.path.join(output_dir, "**", "*.jpg"), recursive=True)
                + glob.glob(os.path.join(output_dir, "**", "*.png"), recursive=True)
                + glob.glob(os.path.join(output_dir, "**", "*.jpeg"), recursive=True)
            )
            if len(images) > 0:
                print(f"SUCCESS: {len(images)} images downloaded")
            else:
                print("WARNING: Download reported success but no images found after waiting")
                print(f"Checked in: {output_dir}")
                print("The download might still be in progress, or there was an issue.")
                print("Try manual download:")
                print("https://universe.roboflow.com/max-mustermann-gmm7j/german-license-plates-hptbz")
                sys.exit(1)
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
