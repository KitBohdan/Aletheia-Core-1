"""Utility for generating a synthetic commands manifest."""

from __future__ import annotations

import csv
import os

os.makedirs("data/synthetic", exist_ok=True)
with open("data/synthetic/commands_manifest.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "filename", "label"])
    for i, (fname, label) in enumerate(
        [
            ("sydity.wav", "сидіти"),
            ("lezhaty.wav", "лежати"),
            ("do_mene.wav", "до мене"),
            ("bark.wav", "голос"),
        ],
        1,
    ):
        writer.writerow([i, fname, label])
print("synthetic manifest generated.")
