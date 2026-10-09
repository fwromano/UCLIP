#!/usr/bin/env python3
"""Check Sim2 label coverage, image hashes, anchors, and independent video replays."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    doc = json.loads(args.manifest.read_text())
    root = Path(doc["data_root"])
    rows = doc["images"]
    by_path = {r["image_path"]: r for r in rows}
    expected = {str(p.relative_to(root)) for p in (root / "sim2_cropped").glob("*/*.png")}
    assert len(by_path) == len(rows) and set(by_path) == expected, "Incomplete or duplicate coverage"
    for row in rows:
        assert row["label_status"] == "verified_source_pixels"
        assert isinstance(row["relative_heading_deg"], int) and 0 <= row["relative_heading_deg"] < 360
        assert row["heading_uncertainty_deg"] in {0, 1, 2}
        assert hashlib.sha256((root / row["image_path"]).read_bytes()).hexdigest() == row["image_sha256"]
    for audit in doc["source_video_audits"]:
        assert audit["unmatched"] == audit["counter_decreases"] == audit["counter_large_jumps"] == 0
    for anchor in doc["reference_crosswalk"]:
        assert anchor["full_set_matches"], "Reference image was not cross-referenced"
        reference_hash = hashlib.sha256((root / anchor["image_path"]).read_bytes()).hexdigest()
        for match in anchor["full_set_matches"]:
            assert by_path[match["image_path"]]["image_sha256"] == reference_hash
            if by_path[match["image_path"]]["vehicle_type"] == "jeep":
                delta = abs((match["relative_heading_deg"] - anchor["target_angle_deg"] + 180) % 360 - 180)
                assert delta <= 7.5, "Jeep front/rear convention or subset match is inconsistent"
    # Known source-counter and reference defects must remain corrected.
    assert by_path["sim2_cropped/White/White_079.png"]["relative_heading_deg"] == 79
    for category in ("Indigo", "Magenta", "White", "Yellow"):
        assert by_path[f"sim2_cropped/{category}/{category}_990.png"]["relative_heading_deg"] == 99
    assert by_path["sim2_cropped/Wolf2/Wolf2_180.png"]["relative_heading_deg"] == 130
    sample = {}
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
        if row["label_method"] != "source_video_counter":
            sample[row["image_path"]] = row
    for group in groups.values():
        for target in range(0, 360, 45):
            row = min(group, key=lambda r: abs((r["relative_heading_deg"]-target+180)%360-180))
            sample[row["image_path"]] = row
        corrected = [r for r in group if r["filename_corrected"]]
        for row in corrected[::max(len(corrected)//8, 1)]:
            sample[row["image_path"]] = row
    caps = {}
    try:
        for row in sample.values():
            video = row["source_video"]
            if video not in caps:
                caps[video] = cv2.VideoCapture(video)
            cap = caps[video]
            cap.set(cv2.CAP_PROP_POS_FRAMES, row["source_frame_index"])
            ok, frame = cap.read()
            assert ok
            x, y, w, h = row["crop_xywh"]
            crop = cv2.imread(str(root / row["image_path"]))
            assert np.array_equal(crop, frame[y:y+h, x:x+w]), row["image_path"]
    finally:
        for cap in caps.values():
            cap.release()
    print(json.dumps({"coverage_and_hashes_checked": len(rows), "source_frames_independently_replayed": len(sample),
                      "reference_images_cross_checked": len(doc["reference_crosswalk"]), "result": "pass"}, indent=2))


if __name__ == "__main__":
    main()
