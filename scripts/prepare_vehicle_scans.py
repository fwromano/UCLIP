#!/usr/bin/env python3
"""Audit/crop the complete MAAS scan ZIP without extracting raw frames."""
import argparse,hashlib,io,json,os,sys,time
from collections import Counter,defaultdict
from pathlib import Path
from zipfile import ZipFile

os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/uclip-vehicle-scans-yolo')
os.environ.setdefault('MPLCONFIGDIR','/tmp/uclip-vehicle-scans-mpl')
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from uclip.studies.vehicle_scans import scan_record,central_yolo_boxes,color_variant_bounds,green_body_bounds,padded_union
from PIL import Image,ImageDraw
import torch
from ultralytics import YOLO


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--yolo-weights',type=Path,required=True);p.add_argument('--crop-overrides',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    overrides=json.loads(a.crop_overrides.read_text()) if a.crop_overrides else {}
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(0)
    model=YOLO(str(a.yolo_weights));a.output.mkdir(parents=True);(a.output/'crops').mkdir()
    with ZipFile(a.archive) as archive:
        names=sorted(n for n in archive.namelist() if n.lower().endswith('.png'))
        rows=[dict(scan_record(n),embedding_row=i) for i,n in enumerate(names)]
        groups=defaultdict(list)
        for r in rows:
            key=('truck',r['elevation_deg'],r['source_azimuth_deg']) if r['vehicle_type']=='truck' else (r['series'],r['elevation_deg'],r['source_azimuth_deg'])
            groups[key].append(r)
        failures=[]
        for number,(key,batch) in enumerate(sorted(groups.items(),key=lambda kv:str(kv[0]))):
            raw=[archive.read(r['source_member']) for r in batch]
            images=[Image.open(io.BytesIO(b)).convert('RGB') for b in raw]
            results=model.predict(images,device='cpu',imgsz=640,conf=.15,verbose=False)
            proposals=[central_yolo_boxes(result,*image.size) for image,result in zip(images,results)]
            if key[0]=='truck':
                mask=color_variant_bounds(images)
                bounds=[mask]+[proposal['bounds'] for candidates in proposals for proposal in candidates]
                method='shared-paint-variants+central-yolo-proposals'
            else:
                mask=green_body_bounds(images[0])
                bounds=[mask]+[proposal['bounds'] for proposal in proposals[0]]
                method='green-body+central-yolo-proposals'
            try:crop=padded_union(bounds,images[0].size)
            except ValueError:crop=None
            for r,b,image,candidates in zip(batch,raw,images,proposals):
                chosen=overrides.get(r['source_member'],crop)
                if chosen is None:failures.append(r['source_member']);continue
                if len(chosen)!=4 or not (0<=chosen[0]<chosen[2]<=image.width and 0<=chosen[1]<chosen[3]<=image.height):
                    raise ValueError('Invalid crop bounds for '+r['source_member'])
                if r['source_member'] in overrides:method_for_row='reviewed-override'
                else:method_for_row=method
                r.update(source_sha256=hashlib.sha256(b).hexdigest(),source_size_px=list(image.size),
                         crop_bounds_xyxy=chosen,crop_method=method_for_row,mask_bounds_xyxy=mask,
                         yolo_proposals=candidates,yolo_class_used_as_truth=False,
                         crop_path=f"crops/{r['embedding_row']:04d}.png")
                image.crop(chosen).save(a.output/r['crop_path'])
                r['crop_sha256']=digest(a.output/r['crop_path'])
            if number%28==0:print(json.dumps({'phase':'crop','groups':number+1,'total_groups':len(groups),'elapsed_sec':time.monotonic()-started}),flush=True)
        if failures:raise ValueError('Missing crop proposals: '+json.dumps(failures))
        # Complete coverage is established from filename tuples, not assumed.
        coverage={s:len({(r['elevation_deg'],r['source_azimuth_deg']) for r in rows if r['series']==s}) for s in sorted({r['series'] for r in rows})}
        assert all(v==168 for v in coverage.values()) and len(rows)==2184
        audit={'schema':'maas.vehicle_scan_crops.v1','source_archive':str(a.archive.resolve()),
               'source_archive_sha256':digest(a.archive),'images':rows,'coverage':coverage,
               'filename_convention':{'orbit_parameter_units':'unconfirmed','elevation':'degrees, inferred from cardinal rendered views',
                   'heading':'relative_heading=(270-source_azimuth)%360; front=0, right-facing side=90, rear=180, left-facing side=270; visually inferred'},
               'crop_policy':{'truck':'same crop for all ten paint variants at a viewpoint','others':'central detector proposals plus connected green body, padded',
                  'detector_classes':'retained for audit only; folder labels define study categories','padding_fraction':.045,'minimum_padding_px':10},
               'yolo':{'weights_sha256':digest(a.yolo_weights),'imgsz':640,'confidence_threshold':.15,'device':'cpu','threads':4},
               'counts':{'images':len(rows),'with_central_yolo_proposal':sum(bool(r['yolo_proposals']) for r in rows),
                         'without_central_yolo_proposal':sum(not r['yolo_proposals'] for r in rows)},
               'elapsed_sec':time.monotonic()-started}
        if a.crop_overrides:
            audit['reviewed_overrides']={'path':str(a.crop_overrides.resolve()),'sha256':digest(a.crop_overrides),
                                         'rows':sum(r['crop_method']=='reviewed-override' for r in rows)}
        (a.output/'index.json').write_text(json.dumps(audit,indent=2)+'\n')
        # Stratified contact sheets, native and crop side-by-side, for review.
        series=sorted(coverage)
        for azimuth in (0,90,180,270):
            sheet=Image.new('RGB',(7*230,len(series)*170),'#102033');draw=ImageDraw.Draw(sheet)
            for y,s in enumerate(series):
                for x,e in enumerate((-5,0,18,36,54,72,90)):
                    r=next(r for r in rows if r['series']==s and r['elevation_deg']==e and r['source_azimuth_deg']==azimuth)
                    original=Image.open(io.BytesIO(archive.read(r['source_member']))).convert('RGB');original.thumbnail((112,130))
                    crop=Image.open(a.output/r['crop_path']).convert('RGB');crop.thumbnail((112,130))
                    sheet.paste(original,(x*230,y*170+25));sheet.paste(crop,(x*230+115,y*170+25))
                    draw.text((x*230+3,y*170+5),f'{s} e={e} a={azimuth}',fill='white')
            sheet.save(a.output/f'crop-review-az{azimuth}.jpg',quality=90)
        print(json.dumps({'phase':'complete','output':str(a.output),'counts':audit['counts'],'elapsed_sec':audit['elapsed_sec']}),flush=True)


if __name__=='__main__':main()
