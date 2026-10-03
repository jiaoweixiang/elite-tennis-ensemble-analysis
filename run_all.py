"""Verify deposited data and regenerate the manuscript's numerical summaries."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    for script in ("verify_data.py", "build_processed.py", "audit_source_mapping.py",
                   "descriptive.py", "reproduce.py", "reproduce_top40.py",
                   "reproduce_tuned_top40.py", "bootstrap_ci.py"):
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "analysis" / script)], cwd=ROOT, check=True)
    print("Numerical reproduction complete; see outputs/", flush=True)


if __name__ == "__main__":
    main()
