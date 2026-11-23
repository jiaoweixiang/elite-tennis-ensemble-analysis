"""Verify the deposited data and regenerate the manuscript's numerical results."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = (
    "verify_data.py",                       # checksums, row counts, source corrections
    "build_processed.py",                   # rebuild model inputs from the six workbooks
    "audit_source_mapping.py",              # list documented source-to-table cell differences
    "check_recording_origins.py",           # Table S11 original values vs model inputs
    "descriptive.py",                       # Table S1, Figure 2
    "reproduce.py",                         # Tables S2-S5, S7; Figures 3-4
    "reproduce_top40.py",                   # Table S6 notes, Figure 5
    "bootstrap_ci.py",                      # Table S3 2025 intervals
    "calibration_diagnostics.py",           # Table S4 calibration intercept and slope
    "return_representation_diagnostics.py", # Table S10 VIF
    "reproduce_revision.py",                # Tables S1/S3 notes, S5-S11, Figure 4b
    "ablation_ranks.py",                    # all-indicator 2025 deletion-refit ranking
)


def main() -> None:
    for script in SCRIPTS:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "analysis" / script)], cwd=ROOT, check=True)
    print("Numerical reproduction complete; see outputs/", flush=True)


if __name__ == "__main__":
    main()
