"""Launch this device's frozen batch-one navigation inference service."""

import sys
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parent))
from run_navigation_service import main  # noqa: E402

if __name__ == "__main__":
    main(DIRECTORY)
