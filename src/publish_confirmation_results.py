"""Publish one completed confirmation run into the tracked paper artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_NAMES = {
    "01_fusion_objectives.png": "11_fusion_objectives.png",
    "02_leave_one_out.png": "12_leave_one_out.png",
    "03_state_intervals.png": "13_state_intervals.png",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError(f"run is not completed: {run_dir}")

    destination = ROOT / "artifacts" / "confirmation_results"
    table_destination = destination / "tables"
    figure_destination = destination / "figures"
    paper_assets = ROOT / "paper" / "assets"
    for directory in (table_destination, figure_destination, paper_assets):
        directory.mkdir(parents=True, exist_ok=True)

    for source in (run_dir / "tables").iterdir():
        if source.suffix.lower() in {".csv", ".json"}:
            shutil.copy2(source, table_destination / source.name)
    for source in (run_dir / "figures").glob("*.png"):
        shutil.copy2(source, figure_destination / source.name)
    for source_name, paper_name in FIGURE_NAMES.items():
        source = run_dir / "figures" / source_name
        shutil.copy2(source, paper_assets / paper_name)

    provenance = {
        "source_run": run_dir.name,
        "source_manifest": str(run_dir / "manifest.json"),
        "published_figures": FIGURE_NAMES,
    }
    (destination / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
