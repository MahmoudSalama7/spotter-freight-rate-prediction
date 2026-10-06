import sys
from pathlib import Path

# Add project root directory to sys.path for test discovery
root_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(root_dir))
