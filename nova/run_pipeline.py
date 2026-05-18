"""
Master pipeline — runs all three steps in sequence.
Called by GitHub Actions monthly, or manually anytime.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from collect_data    import run as collect
from compute_signals import run as compute
from run_model       import run as predict

if __name__ == "__main__":
    print("=" * 50)
    print("NoVA Housing Signal Pipeline")
    print("=" * 50)
    print("\n[1/3] Collecting Google Places data...")
    collect()
    print("\n[2/3] Computing signals...")
    compute()
    print("\n[3/3] Running model...")
    predict()
    print("\nDone.")
