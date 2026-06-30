# -*- coding: utf-8 -*-
"""
Driver: run the four modules in dependency order to reproduce all tables and figures.
Usage: python run_all.py
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    "tennis_workflow_v4_en.py",
    "plot_model_ablation.py",
    "plot_importance_heatmap.py",
    "plot_shap_dependence.py",
]


def main():
    for i, script in enumerate(STEPS, 1):
        print(f"\n===== [{i}/{len(STEPS)}] running {script} =====", flush=True)
        ret = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE)
        if ret.returncode != 0:
            print(f"\n[ABORT] {script} failed (exit {ret.returncode}).")
            sys.exit(ret.returncode)
    print("\n[ALL DONE] All tables and figures reproduced (random_state=42).")


if __name__ == "__main__":
    main()
