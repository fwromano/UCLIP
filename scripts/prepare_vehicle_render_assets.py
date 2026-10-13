#!/usr/bin/env python3
"""Prepare local sphere-scan sprites and compare framing with retained Sim2 crops.

All detector classes are audit metadata; source folder labels own identity.
This does not replace or re-encode the previous CLIP study's image crops.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import time
from zipfile import ZipFile

os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/vehicle-render-yolo')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vehicle-render-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounds(mask, padding=5):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('Empty foreground mask')
    h, w = mask.shape
    return [max(0, int(x.min())-padding), max(0, int(y.min())-padding),
            min(w, int(x.max())+padding+1), min(h, int(y.max())+padding+1)]


def roi(row):
    l, t, r, b = row['crop_bounds_xyxy']
    pad = max(25, round(max(r-l, b-t)*.2))
    return [max(0, l-pad), max(0, t-pad), min(640, r+pad), min(640, b+pad)]


def proposal(result, row, region=None, angle=0):
    if result.masks is None:
        return None
    seed = row.get('mask_bounds_xyxy') or row['crop_bounds_xyxy']
    cx, cy = (seed[0]+seed[2])/2, (seed[1]+seed[3])/2
    candidates = []
    for mask, confidence, cls in zip(result.masks.data, result.boxes.conf, result.boxes.cls):
        local = Image.fromarray((mask.cpu().numpy() > .5).astype(np.uint8)*255)
        if angle:
            local = local.rotate(-angle, expand=True)
        full = np.asarray(local) > 0
        if region:
            translated = np.zeros((640, 640), bool)
            translated[region[1]:region[3], region[0]:region[2]] = full
            full = translated
        box = bounds(full, 0)
        if box[0] <= cx <= box[2] and box[1] <= cy <= box[3] and full.sum() >= 64:
            candidates.append((full, {'confidence': float(confidence), 'class': result.names[int(cls)],
                                     'bounds_xyxy': box}))
    return max(candidates, key=lambda item: item[0].sum()) if candidates else None


def audit_sheets(output, audit, rows, masks, raw, crops, sim2_root):
    audit.mkdir(parents=True, exist_ok=True)
    representative = ['truck/yellow', 'jeep/original', 'wolf/original', 'moose/original']
    for az in (0, 90, 180, 270):
        sheet = Image.new('RGB', (7*360, len(representative)*180), '#102033')
        draw = ImageDraw.Draw(sheet)
        for y, series in enumerate(representative):
            for x, elevation in enumerate((-5, 0, 18, 36, 54, 72, 90)):
                row = next(r for r in rows if r['series'] == series and r['source_azimuth_deg'] == az and r['elevation_deg'] == elevation)
                original = raw(row).copy()
                boxdraw = ImageDraw.Draw(original)
                boxdraw.rectangle(row['crop_bounds_xyxy'], outline='orange', width=3)
                boxdraw.rectangle(row['render_bounds_xyxy'], outline='lime', width=3)
                old = Image.open(crops/row['crop_path']).convert('RGB')
                sprite = Image.open(output/row['asset_path']).convert('RGBA')
                cleaned = Image.new('RGB', sprite.size, '#263b39')
                cleaned.paste(sprite, mask=sprite.getchannel('A'))
                for column, image in enumerate((original, old, cleaned)):
                    image.thumbnail((112, 130))
                    sheet.paste(image, (x*360+column*120, y*180+36))
                draw.text((x*360+3, y*180+3), f'{series} e={elevation} az={az}', fill='white')
                draw.text((x*360+3, y*180+18), row['matte_method'], fill='#b6c9d8')
        sheet.save(audit/f'review-az{az}.jpg', quality=94)
    if sim2_root is not None:
        labels = json.loads((sim2_root/'data/car_sim/heading_labels/heading_labels.json').read_text())['images']
        sheet = Image.new('RGB', (8*350, 3*220), '#102033')
        draw = ImageDraw.Draw(sheet)
        matched = []
        for y, (vehicle, color) in enumerate([('jeep', 'Yellow'), ('wolf', None), ('moose', None)]):
            legacy_type = {'wolf': 'wolf2', 'moose': 'moose3'}.get(vehicle, vehicle)
            candidates = [r for r in labels if r['vehicle_type'] == legacy_type and (color is None or r['category'] == color)]
            for x, heading in enumerate(range(0, 360, 45)):
                old = min(candidates, key=lambda r: abs((r['relative_heading_deg']-heading+180)%360-180))
                series = vehicle+'/original'
                new = min((r for r in rows if r['series'] == series and r['elevation_deg'] == 36),
                          key=lambda r: abs((r['relative_heading_deg']-heading+180)%360-180))
                a = Image.open(sim2_root/'data/car_sim'/old['image_path']).convert('RGB')
                b = Image.open(output/new['asset_path']).convert('RGBA')
                bg = Image.new('RGB', b.size, '#263b39'); bg.paste(b, mask=b.getchannel('A'))
                for column, image in enumerate((a, bg)):
                    image.thumbnail((165, 170)); sheet.paste(image, (x*350+column*175, y*220+40))
                draw.text((x*350+3, y*220+3), f'{vehicle} heading {heading} deg', fill='white')
                draw.text((x*350+3, y*220+20), 'Sim2 | scans at 36 deg elevation', fill='#b6c9d8')
                matched.append({'sim2': old['image_path'], 'sim2_sha256': old['image_sha256'],
                                'scan_source': new['source_member'], 'relative_heading_deg': heading})
        sheet.save(audit/'sim2-versus-scans.jpg', quality=94)
        (audit/'sim2-comparison.json').write_text(json.dumps({'pairs': matched, 'limits': 'Headings are matched; Sim2 elevation is uncalibrated, so elevation is not asserted to match.'}, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('archive', 'crops', 'weights', 'output', 'audit'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--sim2-root', type=Path)
    parser.add_argument('--recover-index', action='store_true', help='Index already-written native RGB/alpha sprites by exact source-pixel matching after an interrupted report export.')
    args = parser.parse_args()
    if args.output.exists() and not args.recover_index:
        raise FileExistsError('Use a new output directory; prepared assets are retained.')
    started = time.monotonic()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.manual_seed(0)
    model = YOLO(str(args.weights))
    original = json.loads((args.crops/'index.json').read_text())
    rows = [dict(r) for r in original['images']]
    groups = defaultdict(list)
    for row in rows:
        key = (row['vehicle_type'], row['elevation_deg'], row['source_azimuth_deg'])
        groups[key].append(row)
    selected = [next((r for r in group if r['series'] == 'truck/yellow'), group[0]) for group in groups.values()]
    masks, methods, proposals, full_results = {}, {}, {}, {}
    args.output.mkdir(parents=True, exist_ok=True); (args.output/'sprites').mkdir(exist_ok=True)
    with ZipFile(args.archive) as archive:
        def raw(row):
            data = archive.read(row['source_member'])
            if hashlib.sha256(data).hexdigest() != row['source_sha256']:
                raise ValueError('Source scan changed')
            return Image.open(io.BytesIO(data)).convert('RGB')
        if args.recover_index:
            for reference in selected:
                key = reference['embedding_row']
                sprite = Image.open(args.output/f'sprites/{key:04d}.png').convert('RGBA')
                original_rgb = np.asarray(raw(reference))
                rgb = np.asarray(sprite)[:, :, :3].copy()
                scores = cv2.matchTemplate(original_rgb, rgb, cv2.TM_SQDIFF_NORMED)
                _, _, origin, _ = cv2.minMaxLoc(scores)
                x, y = origin
                if not np.array_equal(original_rgb[y:y+sprite.height, x:x+sprite.width], rgb):
                    raise ValueError('Recovered sprite is not an exact native source crop')
                full = np.zeros((640, 640), bool)
                full[y:y+sprite.height, x:x+sprite.width] = np.asarray(sprite)[:, :, 3] > 0
                masks[key] = full
                expected_context = np.zeros_like(full)
                l,t,r,b = roi(reference); expected_context[t:b,l:r] = True
                methods[key] = 'opaque-context-fallback' if np.array_equal(full, expected_context) else 'segmented-silhouette-recovered'
                proposals[key] = {'provenance': 'Exact native RGB and saved alpha recovered after report export interruption; individual detector/retry labels were not retained.'}
                full_results[key] = None
        for begin in range(0, 0 if args.recover_index else len(selected), 8):
            batch = selected[begin:begin+8]
            results = model.predict([raw(r) for r in batch], device='cpu', imgsz=640,
                                    conf=.15, verbose=False, retina_masks=True)
            for row, result in zip(batch, results):
                chosen = proposal(result, row)
                if chosen:
                    mask, info = chosen
                    masks[row['embedding_row']] = mask; methods[row['embedding_row']] = 'yolo-seg-full'
                    proposals[row['embedding_row']] = info
                full_results[row['embedding_row']] = chosen[1] if chosen else None
            if begin % 64 == 0:
                print(json.dumps({'phase': 'full segmentation', 'views': begin+len(batch), 'total': len(selected)}), flush=True)
        for angle in (() if args.recover_index else (0, 90, 180, 270)):
            missing = [r for r in selected if r['embedding_row'] not in masks]
            for begin in range(0, len(missing), 8):
                batch = missing[begin:begin+8]; regions = [roi(r) for r in batch]
                results = model.predict([raw(r).crop(region).rotate(angle, expand=True) for r, region in zip(batch, regions)],
                                        device='cpu', imgsz=640, conf=.15, verbose=False, retina_masks=True)
                for row, region, result in zip(batch, regions, results):
                    chosen = proposal(result, row, region, angle)
                    if chosen:
                        masks[row['embedding_row']], proposals[row['embedding_row']] = chosen
                        methods[row['embedding_row']] = f'yolo-seg-roi-rotate-{angle}'
            print(json.dumps({'phase': 'ROI retry', 'rotation': angle, 'remaining': len([r for r in selected if r['embedding_row'] not in masks])}), flush=True)
        # At 90° yaw rotates the image, not the viewing ray. Transfer an actual
        # same-model pole mask in the 640x640 source plane; source RGB is retained.
        for row in selected:
            key = row['embedding_row']
            if key in masks or row['elevation_deg'] != 90:
                continue
            donor = next((r for r in selected if r['vehicle_type'] == row['vehicle_type'] and r['elevation_deg'] == 90 and r['embedding_row'] in masks), None)
            if donor:
                rotation = donor['source_azimuth_deg']-row['source_azimuth_deg']
                masks[key] = np.asarray(Image.fromarray(masks[donor['embedding_row']].astype(np.uint8)*255).rotate(rotation)) > 0
                methods[key] = 'pole-rotated-mask'; proposals[key] = {'donor_source_member': donor['source_member'], 'rotation_deg': rotation}
        for group in groups.values():
            reference = next((r for r in group if r['series'] == 'truck/yellow'), group[0])
            key = reference['embedding_row']
            mask = masks.get(key)
            if mask is None:
                # Explicit RGB-context fallback preserves the whole body rather
                # than claiming a silhouette where segmentation failed.
                mask = np.zeros((640, 640), bool)
                l,t,r,b = roi(reference); mask[t:b,l:r] = True
                methods[key] = 'opaque-context-fallback'; proposals[key] = None
            crop_bounds = bounds(mask)
            for row in group:
                image = raw(row)
                sprite = Image.fromarray(np.dstack([np.asarray(image), mask.astype(np.uint8)*255]), 'RGBA').crop(crop_bounds)
                path = f'sprites/{row["embedding_row"]:04d}.png'; sprite.save(args.output/path)
                old_bounds = row['crop_bounds_xyxy']; old_area = (old_bounds[2]-old_bounds[0])*(old_bounds[3]-old_bounds[1])
                row.update(asset_path=path, asset_sha256=sha(args.output/path), render_bounds_xyxy=crop_bounds,
                           matte_method=methods[key], mask_reference=reference['source_member'],
                           segmentation=proposals[key], full_frame_segmentation=full_results[key],
                           render_area_vs_previous=(sprite.width*sprite.height)/old_area,
                           foreground_fraction=float(mask.sum())/(sprite.width*sprite.height))
    counts = Counter(r['matte_method'] for r in rows)
    result = {'schema': 'scene-composer.vehicle-assets.v1', 'images': rows,
              'source_archive_sha256': original['source_archive_sha256'],
              'source_crop_index_sha256': sha(args.crops/'index.json'),
              'generator_sha256': sha(Path(__file__)), 'weights_sha256': sha(args.weights),
              'segmentation': {'model': 'yolov8n-seg', 'imgsz': 640, 'confidence': .15, 'classes_used_as_truth': False,
                               'retina_masks': True, 'device': 'cpu', 'threads': 4},
              'padding_px': 5, 'mask_counts': dict(counts), 'independent_geometry_views': len(selected),
              'truck_policy': 'Shared yellow-truck mask across all ten paint variants at the same view.',
              'limits': ['Uncalibrated source camera/physical dimensions.', 'Masks are proposals, not certified full-body segmentation.',
                         'Failed silhouettes retain explicit opaque source context.', 'Original CLIP cache and its padded crops are unchanged.'],
              'elapsed_sec': time.monotonic()-started}
    (args.output/'index.json').write_text(json.dumps(result, indent=2)+'\n')
    # Persist the complete index before any optional visual/comparison export.
    with ZipFile(args.archive) as archive:
        def raw(row):
            return Image.open(io.BytesIO(archive.read(row['source_member']))).convert('RGB')
        audit_sheets(args.output, args.audit, rows, masks, raw, args.crops, args.sim2_root)
    (args.audit/'asset-summary.json').write_text(json.dumps({k:v for k,v in result.items() if k!='images'}, indent=2)+'\n')
    print(json.dumps({'phase': 'complete', 'rows': len(rows), 'mask_counts': dict(counts), 'elapsed_sec': result['elapsed_sec']}), flush=True)


if __name__ == '__main__':
    main()
