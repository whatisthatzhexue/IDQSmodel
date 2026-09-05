from pathlib import Path
import sys


SCORING_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SCORING_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
