#!/usr/bin/env python3
"""Recover Sim2 crop headings by matching original video pixels and reading its counter.

No model weights or network are needed. Original images are never modified.
"""
from __future__ import annotations

import argparse
import ast
import bisect
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_legacy_decoder(path):
    names = {"AngleExtractionConfig", "pad_to_square", "create_digit_templates", "classify_digit", "extract_angle"}
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("Original extraction decoder has changed")
    ns = {"cv2": cv2, "np": np}
    exec("from dataclasses import dataclass\nfrom typing import Dict, Optional", ns)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    config = ns["AngleExtractionConfig"]()
    templates = ns["create_digit_templates"](config)
    return lambda frame: ns["extract_angle"](frame, config, templates)


def digit_glyphs(frame, category):
    # The Wolf capture uses larger text, including a wider "Angle:" prefix.
    start = 130 if category == "Wolf2" else 98
    bottom = 58 if category == "Wolf2" else 38
    gray = cv2.cvtColor(frame[:bottom, :235], cv2.COLOR_BGR2GRAY)
    binary = (gray > 200).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x >= start and h >= 10 and w >= 3 and w * h >= 40:
            boxes.append((x, y, w, h))
    result = []
    for x, y, w, h in sorted(boxes):
        glyph = binary[y:y+h, x:x+w]
        result.append(cv2.resize(glyph, (24, 32), interpolation=cv2.INTER_AREA) > 128)
    return result


def make_counter_templates(video):
    # These visible source counters were manually read from the source-video audit.
    training = {0: "17", 40: "21", 80: "24", 100: "25", 180: "32", 220: "35",
                260: "39", 600: "67", 800: "84", 1000: "100", 1090: "108", 4079: "358"}
    cap = cv2.VideoCapture(str(video))
    templates = {}
    try:
        for frame_index, text in training.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot read training frame {frame_index}")
            glyphs = digit_glyphs(frame, "Indigo")
            if len(glyphs) != len(text):
                raise ValueError(f"Unexpected counter segmentation at training frame {frame_index}")
            for digit, glyph in zip(text, glyphs):
                templates.setdefault(digit, []).append(glyph)
    finally:
        cap.release()
    assert set(templates) == set("0123456789")
    return templates


def read_counter(frame, category, templates):
    glyphs = digit_glyphs(frame, category)
    if not 1 <= len(glyphs) <= 3:
        return None, None
    digits, errors = [], []
    for glyph in glyphs:
        candidates = [(min(float(np.mean(glyph != sample)) for sample in samples), digit)
                      for digit, samples in templates.items()]
        error, digit = min(candidates)
        if error > 0.15:
            return None, error
        digits.append(digit)
        errors.append(error)
    angle = int("".join(digits))
    return (angle if 0 <= angle <= 360 else None), max(errors)


def locate_exact_crop(frame, crop):
    h, w = crop.shape[:2]
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Flat paint patches have many identical locations, especially on White.
    patches = [(float(gray_crop[y:y+24,x:x+24].std()), y, x)
               for y in np.linspace(0, h-24, 6, dtype=int)
               for x in np.linspace(0, w-24, 6, dtype=int)]
    for _, cy, cx in sorted(patches, reverse=True)[:3]:
        patch = gray_crop[cy:cy+24, cx:cx+24]
        scores = cv2.matchTemplate(gray_frame, patch, cv2.TM_SQDIFF)
        _, _, position, _ = cv2.minMaxLoc(scores)
        x, y = position[0] - cx, position[1] - cy
        if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            continue
        if np.array_equal(crop, frame[y:y+h, x:x+w]):
            return [int(x), int(y), w, h]
    return None


def write_review(document, output):
    payload = json.dumps(document["images"], separators=(",", ":")).replace("<", "\\u003c")
    template = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Sim2 heading labels</title>
<style>body{font:16px system-ui,sans-serif;background:#101722;color:#ecf1f8;margin:0}main{max-width:1100px;margin:auto;padding:28px}
h1{margin-bottom:8px}p{color:#b8c7d8;line-height:1.5}a{color:#8bc8ff}label{display:inline-block;margin:12px 20px 12px 0}
select,input,button{font:inherit}input[type=range]{width:240px;vertical-align:middle}button,select{padding:8px;background:#233247;color:white;border:1px solid #526581;border-radius:6px}
.focus{display:grid;grid-template-columns:minmax(0,2fr) minmax(220px,1fr);gap:24px;background:#192537;padding:18px;border-radius:10px}
.focus img{max-width:100%;max-height:500px;object-fit:contain;justify-self:center}.meta{overflow-wrap:anywhere;line-height:1.6}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:20px}.card{background:#192537;padding:10px;border:0;color:white;cursor:pointer;text-align:left}.card img{width:100%;height:140px;object-fit:contain}.card span{display:block;margin-top:8px}
@media(max-width:700px){.focus{grid-template-columns:1fr}.grid{grid-template-columns:repeat(2,1fr)}}
</style><main><h1>Sim2 relative heading labels</h1>
<p>0° = front facing the camera · 90° = nose toward image right · 180° = rear · 270° = nose toward image left.<br>
Angles are relative viewing azimuths from the source-video counter. They are not compass headings. Original images retain their filenames.</p>
<p><a href="heading_labels.csv">CSV labels</a> · <a href="heading_labels.json">JSON labels and provenance</a> · <a href="README.md">Method and limitations</a></p>
<label>Category <select id="category"></select></label><label>Requested heading <input id="angle" type="range" min="0" max="359" value="0"> <output id="target">0°</output></label>
<button id="previous">Previous image</button> <button id="next">Next image</button>
<p id="count"></p><section class="focus"><img id="image" alt="Selected Sim2 image"><div class="meta" id="metadata"></div></section>
<div class="grid" id="grid"></div></main><script>
const rows=__ROWS__;
const category=document.querySelector('#category'),slider=document.querySelector('#angle');let current=[];let selected=0;
const distance=(a,b)=>Math.abs(((a-b+540)%360)-180);
const imageUrl=row=>'../'+row.image_path.split('/').map(encodeURIComponent).join('/');
for(const name of [...new Set(rows.map(row=>row.category))]){const o=document.createElement('option');o.value=name;o.textContent=name;category.append(o);}
function closest(angle){return current.reduce((best,row,i)=>distance(row.relative_heading_deg,angle)<distance(current[best].relative_heading_deg,angle)?i:best,0);}
function show(index){selected=(index+current.length)%current.length;const row=current[selected];document.querySelector('#image').src=imageUrl(row);document.querySelector('#image').alt=row.category+' at '+row.relative_heading_deg+' degrees';
const metadata=document.querySelector('#metadata');metadata.replaceChildren();for(const text of [row.relative_heading_deg+'° relative heading',row.image_path,'Vehicle: '+row.vehicle_type,'Method: '+row.label_method,'Label uncertainty: '+row.heading_uncertainty_deg+'°','Source frame: '+row.source_frame_index+' ('+row.source_time_sec+' s)','Filename angle: '+row.filename_angle+(row.filename_corrected?' — corrected in manifest':''),row.label_status]){const p=document.createElement('p');p.textContent=text;metadata.append(p);}}
function load(){current=rows.filter(row=>row.category===category.value&&row.relative_heading_deg!==null).sort((a,b)=>a.relative_heading_deg-b.relative_heading_deg||a.image_path.localeCompare(b.image_path));document.querySelector('#count').textContent=current.length+' labeled images in '+category.value+'. Nearest available view is shown for each requested angle.';document.querySelector('#grid').replaceChildren();for(let angle=0;angle<360;angle+=45){const i=closest(angle),row=current[i],card=document.createElement('button'),img=document.createElement('img'),caption=document.createElement('span');card.className='card';img.src=imageUrl(row);img.loading='lazy';img.alt=category.value+' at '+row.relative_heading_deg+' degrees';caption.textContent=angle+'° target → '+row.relative_heading_deg+'° image';card.append(img,caption);card.onclick=()=>{slider.value=angle;document.querySelector('#target').textContent=angle+'°';show(i)};document.querySelector('#grid').append(card);}show(closest(Number(slider.value)));}
category.onchange=load;slider.oninput=()=>{document.querySelector('#target').textContent=slider.value+'°';show(closest(Number(slider.value)))};document.querySelector('#previous').onclick=()=>show(selected-1);document.querySelector('#next').onclick=()=>show(selected+1);load();
</script></html>"""
    (output / "heading_review.html").write_text(template.replace("__ROWS__", payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--legacy-extractor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--reviewed-labels", type=Path, help="Hash-bound human reviews of difficult source counters")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    legacy = load_legacy_decoder(args.legacy_extractor)
    templates = make_counter_templates(args.video_root / "Indigo.mp4")
    rows, video_audits = [], []
    folders = sorted((args.data_root / "sim2_cropped").iterdir())
    for folder in folders:
        if not folder.is_dir() or (args.categories and folder.name not in args.categories):
            continue
        files = sorted(folder.glob("*.png"))
        if not files:
            continue
        category = folder.name
        video = args.video_root / f"{category}.mp4"
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f"Cannot open {video}")
        pending = {int(p.stem.rsplit("_", 1)[1]): p for p in files}
        images = {}
        index, observed, bad_counter, match_attempts = 0, [], 0, 0
        matched_rows = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                source_angle, counter_error = read_counter(frame, category, templates)
                if source_angle is None:
                    bad_counter += 1
                else:
                    observed.append((index, source_angle))
                legacy_angle = legacy(frame)
                if legacy_angle in pending:
                    path = pending[legacy_angle]
                    if legacy_angle not in images:
                        images[legacy_angle] = cv2.imread(str(path))
                    match_attempts += 1
                    rect = locate_exact_crop(frame, images[legacy_angle])
                    if rect is not None:
                        row = {
                            "image_path": str(path.relative_to(args.data_root)),
                            "category": category,
                            "vehicle_type": "jeep" if category in {"Indigo", "Magenta", "White", "Yellow"} else category.lower(),
                            "filename_angle": legacy_angle,
                            "relative_heading_deg": source_angle % 360 if source_angle is not None else None,
                            "source_counter_deg": source_angle,
                            "label_status": "verified_source_pixels" if source_angle is not None else "counter_review_required",
                            "source_video": str(video),
                            "source_frame_index": index,
                            "source_time_sec": round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 6),
                            "crop_xywh": rect,
                            "counter_glyph_error": counter_error,
                            "label_method": "source_video_counter" if source_angle is not None else None,
                            "heading_uncertainty_deg": 0 if source_angle is not None else None,
                            "image_sha256": sha256(path),
                            "filename_corrected": source_angle is not None and legacy_angle != source_angle,
                        }
                        rows.append(row)
                        matched_rows.append(row)
                        del pending[legacy_angle]
                        del images[legacy_angle]
                index += 1
        finally:
            cap.release()
        # A few original video overlays contain corrupt values, e.g. "990".
        # Bound temporal repairs by neighboring valid counters; retain uncertainty.
        indices = [i for i, _ in observed]
        for row in matched_rows:
            if row["relative_heading_deg"] is not None:
                continue
            position = bisect.bisect_left(indices, row["source_frame_index"])
            if position == 0 or position == len(observed):
                continue
            (left_i, left_a), (right_i, right_a) = observed[position-1:position+1]
            delta = (right_a - left_a) % 360
            if right_i-left_i > 36 or delta > 3:
                continue
            fraction = (row["source_frame_index"]-left_i)/(right_i-left_i)
            angle = int(round(left_a + fraction*delta)) % 360
            row.update(relative_heading_deg=angle, label_status="verified_source_pixels",
                       label_method="neighboring_source_counters", heading_uncertainty_deg=max(delta, 1),
                       repair_counter_bracket=[[left_i,left_a],[right_i,right_a]],
                       filename_corrected=row["filename_angle"] != angle)
        # Counter continuity is independent of the legacy filename decoder.
        decreases = [(a, b) for a, b in zip(observed, observed[1:]) if b[1] < a[1] and not (a[1] >= 350 and b[1] <= 10)]
        jumps = [(a, b) for a, b in zip(observed, observed[1:]) if ((b[1]-a[1]) % 360) > 3]
        for legacy_angle, path in pending.items():
            rows.append({"image_path": str(path.relative_to(args.data_root)), "category": category,
                         "vehicle_type": "jeep" if category in {"Indigo", "Magenta", "White", "Yellow"} else category.lower(),
                         "filename_angle": legacy_angle, "relative_heading_deg": None,
                         "label_status": "source_match_review_required", "image_sha256": sha256(path)})
        audit = {"category": category, "video": str(video), "video_sha256": sha256(video),
                 "frames": index, "crop_count": len(files), "matched": len(matched_rows),
                 "unmatched": len(pending), "counter_unreadable_frames": bad_counter,
                 "counter_decreases": len(decreases), "counter_large_jumps": len(jumps),
                 "counter_problem_samples": (decreases + jumps)[:10], "match_attempts": match_attempts,
                 "filename_corrections": sum(bool(r.get("filename_corrected")) for r in matched_rows)}
        video_audits.append(audit)
        print(json.dumps(audit), flush=True)

    if args.reviewed_labels:
        reviews = json.loads(args.reviewed_labels.read_text())
        by_path = {row["image_path"]: row for row in rows}
        for review in reviews["images"]:
            row = by_path.get(review["image_path"])
            if row is None:
                if args.categories:
                    continue
                raise ValueError(f"Reviewed image missing: {review['image_path']}")
            if row["image_sha256"] != review["image_sha256"] or row.get("source_frame_index") != review["source_frame_index"]:
                raise ValueError(f"Reviewed source changed: {review['image_path']}")
            angle = review["relative_heading_deg"]
            if not isinstance(angle, int) or not 0 <= angle < 360:
                raise ValueError("Reviewed angle must be an integer in [0,360)")
            row.update(automatic_relative_heading_deg=row["relative_heading_deg"], relative_heading_deg=angle,
                       source_counter_deg=review.get("source_counter_deg"), source_counter_text=review["source_counter_text"],
                       label_method=review["label_method"], heading_uncertainty_deg=review["heading_uncertainty_deg"],
                       review_note=review["review_note"], filename_corrected=row["filename_angle"] != angle)
        for audit in video_audits:
            audit["filename_corrections"] = sum(bool(r.get("filename_corrected")) for r in rows if r["category"] == audit["category"])
    rows.sort(key=lambda row: row["image_path"])
    by_hash = {}
    for row in rows:
        by_hash.setdefault(row["image_sha256"], []).append(row)
    anchors = []
    for path in sorted((args.data_root / "sim2_cropped_45deg").glob("*/*.png")):
        if args.categories and path.parent.name not in args.categories:
            continue
        matches = by_hash.get(sha256(path), [])
        anchors.append({"image_path": str(path.relative_to(args.data_root)), "target_angle_deg": int(path.stem.rsplit("_", 1)[1]),
                        "full_set_matches": [{"image_path": row["image_path"], "relative_heading_deg": row.get("relative_heading_deg")}
                                             for row in matches]})
    document = {"schema": "sim2.relative_heading_labels.v1", "data_root": str(args.data_root),
                "convention": {"zero": "front facing camera", "180": "rear facing camera",
                               "90": "vehicle nose points to image right", "270": "vehicle nose points to image left",
                               "range": "[0, 360)", "meaning": "relative viewing azimuth; not compass heading",
                               "precision": "integer angle displayed by original renderer",
                               "visual_calibration": "Jeep cardinal views checked against the 45-degree reference set"},
                "legacy_extractor": {"path": str(args.legacy_extractor), "sha256": sha256(args.legacy_extractor)},
                "reviewed_labels_sha256": sha256(args.reviewed_labels) if args.reviewed_labels else None,
                "images": rows, "source_video_audits": video_audits, "reference_crosswalk": anchors}
    (args.output / "heading_labels.json").write_text(json.dumps(document, indent=2) + "\n")
    fields = ["image_path", "category", "vehicle_type", "relative_heading_deg", "source_counter_deg", "filename_angle",
              "filename_corrected", "label_status", "label_method", "heading_uncertainty_deg", "source_video", "source_frame_index", "source_time_sec", "image_sha256"]
    with (args.output / "heading_labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"images": len(rows), "label_status": dict(Counter(r["label_status"] for r in rows)),
               "filename_corrections": sum(bool(r.get("filename_corrected")) for r in rows),
               "label_methods": dict(Counter(r.get("label_method") for r in rows)),
               "anchors": len(anchors), "anchors_matched": sum(bool(a["full_set_matches"]) for a in anchors)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_review(document, args.output)
    print(json.dumps(summary), flush=True)
    if any(row["label_status"] != "verified_source_pixels" for row in rows) or any(a["counter_decreases"] or a["counter_large_jumps"] for a in video_audits):
        raise SystemExit("Review required; uncertain labels retained explicitly")


if __name__ == "__main__":
    main()
