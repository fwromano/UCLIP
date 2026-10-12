#!/usr/bin/env python3
"""Publish a bounded Sim2/sphere explorer and the numerical study report."""
import argparse,base64,hashlib,json,os,shutil,sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/uclip-vehicle-scans-mpl')
import numpy as np
from PIL import Image
from build_vehicle_scan_study import digest,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--crops',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--camera',type=Path,required=True);p.add_argument('--sim2-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--refresh',action='store_true')
    a=p.parse_args()
    if a.output.exists() and not a.refresh:raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'previews').mkdir(exist_ok=True)
    source=Path(__file__).resolve().parents[1]
    projection=json.loads((a.study/'projection.json').read_text());evaluation=json.loads((a.study/'evaluation.json').read_text())
    audit=json.loads((a.crops/'index.json').read_text());camera=json.loads((a.camera/'index.json').read_text())
    if evaluation['source_crop_index_sha256']!=digest(a.crops/'index.json') or camera['crop_index_sha256']!=digest(a.crops/'index.json'):
        raise ValueError('Crop/study/camera association mismatch')
    old=np.load(a.sim2_root/'data/car_sim/clip_embeddings/embeddings.npy',allow_pickle=False)
    new=np.load(a.study/'cropped_embeddings.npy',allow_pickle=False)
    old_colors={'Indigo':'#9d86ff','Magenta':'#f56dbd','White':'#d7e5f5','Yellow':'#f8ca56'}
    new_colors={'black':'#87939c','blue':'#477eff','gray':'#bfc4cb','green':'#51dc7d','indigo':'#bb83ed','orange':'#ff9d52','red':'#ef7583','violet':'#d88aef','white':'#e6edf6','yellow':'#f8d35c','original':'#80b8a2'}
    rows=[];colors={};series_meta={};vectors=[]
    for dataset,records in [('sim2',projection['sim2']),('scans',projection['scans'])]:
        for r in records:
            if dataset=='sim2':
                category='Sim2 · Jeep / '+r['category'];color=old_colors[r['category']]
                path=a.sim2_root/'data/car_sim'/r['image_path'];heading=r['relative_heading_deg'];elevation=None
                vector=old[r['embedding_row']]
            else:
                category='Scans · '+r['series'];color=new_colors[r['color']]
                path=a.crops/r['crop_path'];heading=r['relative_heading_deg'];elevation=r['elevation_deg']
                vector=new[r['embedding_row']]
            colors[category]=color;series_meta[category]={'dataset':dataset,'step':5 if dataset=='sim2' else 15}
            rows.append({'embedding_row':len(rows),'source_embedding_row':r['embedding_row'],'dataset':dataset,
                         'category':category,'heading':heading,'elevation':elevation,'coords':r['coords'],
                         'image_path':r.get('image_path',r.get('source_member')),'source_sha256':r.get('image_sha256',r.get('source_sha256')),
                         'crop_sha256':r.get('crop_sha256',r.get('image_sha256')),'uncertainty':r.get('heading_uncertainty_deg',0),
                         '_local_path':path})
            vectors.append(vector)
    assert len(rows)==3588 and len(colors)==17
    vectors=np.stack(vectors);references={};sheets=[]
    for category in colors:
        group=[r for r in rows if r['category']==category]
        ref=min(group,key=lambda r:(abs((r['heading']+180)%360-180),abs((r['elevation'] if r['elevation'] is not None else 54)-54),r['embedding_row']))
        references[category]=ref['embedding_row']
        ordered=sorted(group,key=lambda r:(r['elevation'] or 0,r['heading'],r['embedding_row']))
        name_prefix=category.lower().replace(' · ','-').replace(' / ','-').replace('/','-').replace(' ','-')
        for begin in range(0,len(ordered),64):
            batch=ordered[begin:begin+64];name=f'previews/{name_prefix}-{begin//64:02d}.webp'
            sheet=Image.new('RGB',(2048,176*((len(batch)+7)//8)),'#131c29')
            for cell,row in enumerate(batch):
                if digest(row['_local_path'])!=row['crop_sha256']:raise ValueError('Preview source changed')
                image=Image.open(row['_local_path']).convert('RGB');image.thumbnail((256,176))
                sheet.paste(image,((cell%8)*256+(256-image.width)//2,(cell//8)*176+(176-image.height)//2))
                row.update(preview_sheet=name,preview_cell=cell,cos_to_front=float(vectors[row['embedding_row']]@vectors[ref['embedding_row']]))
            sheet.save(a.output/name,format='WEBP',quality=80,method=4);sheets.append(name)
    for row in rows:row.pop('_local_path')
    payload={'rows':rows,'colors':colors,'seriesMeta':series_meta,'elevations':[-5,0,18,36,54,72,90],
             'defaultSeries':'Scans · truck/yellow','references':references,'ratio':projection['explained_variance_ratio'],
             'model':evaluation['model'],'cameraSamples':camera['observations'],
             'imageBase':'https://raw.githubusercontent.com/fwromano/UCLIP/77dee02fe81b28ba2f2c33e46a9a905e4c7498b5/data/car_sim/',
             'labelsUrl':'https://fwromano.github.io/datasets/sim2-headings/','cacheUrl':'evaluation.json','methodsUrl':'index.html'}
    template=(source/'src/uclip/viz/vehicle_scan_orbit.html').read_text().replace('__ORBIT_JS__',(source/'src/uclip/viz/vehicle_scan_orbit.js').read_text())
    def render(data):return template.replace('__ORBIT_DATA__',json.dumps(data,separators=(',',':'),allow_nan=False).replace('</','<\\/'))
    (a.output/'embedding.html').write_text(render(payload))
    offline=dict(payload,atlases={name:'data:image/webp;base64,'+base64.b64encode((a.output/name).read_bytes()).decode() for name in sheets})
    # Camera crops in offline viewers are inline too.
    offline['cameraSamples']=[dict(r,offline_image='data:image/png;base64,'+base64.b64encode((a.camera/r['crop_path']).read_bytes()).decode()) for r in camera['observations']]
    offline_html=render(offline).replace("'camera/'+row.crop_path","row.offline_image||('camera/'+row.crop_path)")
    (a.output/'embedding_offline.html').write_text(offline_html)
    for name in ['evaluation.json','projection.json','raw_embeddings.npy','cropped_embeddings.npy','resolution_embeddings.npy','elevation_holdout.json']:
        shutil.copyfile(a.study/name,a.output/name)
    clean_audit=dict(audit,source_archive=Path(audit['source_archive']).name)
    if 'reviewed_overrides' in clean_audit:
        clean_audit['reviewed_overrides']=dict(clean_audit['reviewed_overrides'],path=Path(clean_audit['reviewed_overrides']['path']).name)
    write(a.output/'crop_index.json',clean_audit)
    shutil.copytree(a.camera,a.output/'camera',dirs_exist_ok=True)
    for name in ['crop-review-az0.jpg','crop-review-az90.jpg','crop-review-az180.jpg','crop-review-az270.jpg']:
        shutil.copyfile(a.crops/name,a.output/name)
    if 'reviewed_overrides' in audit:
        shutil.copyfile(audit['reviewed_overrides']['path'],a.output/'crop-overrides.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    native=evaluation['native_held_out_azimuth']
    axes[0].bar(['Full frame','Padded crop'],[native['raw']['accuracy']*100,native['cropped']['accuracy']*100],color=['#889bbd','#5389d7'])
    axes[0].set(ylim=(0,105),ylabel='13-series classification accuracy (%)',title='Held-out azimuths: 1,092 queries')
    for key,label in [('native_gallery','Original crop references'),('matched_resolution_gallery','Matched-resolution references')]:
        axes[1].plot([r['crop_width_px'] for r in evaluation['resolution']],[r[key]['accuracy']*100 for r in evaluation['resolution']],marker='o',label=label)
    axes[1].set(xscale='log',ylim=(0,105),xlabel='Query crop width (pixels)',ylabel='Classification accuracy (%)',title='91 queries / 91 separate reference views')
    axes[1].set_xticks([r['crop_width_px'] for r in evaluation['resolution']], [str(r['crop_width_px']) for r in evaluation['resolution']]);axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
    fig.savefig(a.output/'study.png',dpi=170);fig.savefig(a.output/'study.svg');plt.close(fig)
    svg=a.output/'study.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    table='\n'.join(f"| {r['crop_width_px']} | {r['native_gallery']['accuracy']*100:.1f}% | {r['matched_resolution_gallery']['accuracy']*100:.1f}% |" for r in evaluation['resolution'])
    report=f'''# Sim2 + MAAS vehicle sphere appearance study — 2026-10-12

[Interactive embedding explorer](embedding.html) · [Offline explorer](embedding_offline.html) · [Existing Sim2 explorer](https://fwromano.github.io/datasets/sim2-headings/embedding.html)

The new archive contains **2,184 640×640 rendered frames**, arranged as 13 series: ten paint variants of the truck, and one original Jeep, Wolf and Moose. Each has 24 azimuths at each of seven elevations: −5, 0, 18, 36, 54, 72 and 90 degrees. This is hemisphere coverage plus one slightly below-horizontal ring, not a complete lower sphere.

## Cropping and labels

The uncropped frames contain sky and ground grid. CPU YOLOv8n generated central bounding-box proposals for {audit['counts']['with_central_yolo_proposal']} frames; **{audit['counts']['without_central_yolo_proposal']} had no such proposal**. Its class labels were frequently unsuitable for these renderings (for example overhead vehicles labeled cell phone); detector classes never replace folder labels. Padded crop proposals combine detector geometry with shared paint differences for trucks and a connected green-body mask for other models. All ten truck colors share a crop rectangle at a viewpoint. Thirty images (three truck viewpoints across all colors) use reviewed overrides after paint differences picked up sparse background variation. Four cardinal contact sheets cover 364 source/crop pairs; crop proposals are not manually certified across every frame.

Filename azimuth 270 shows the front, 90 the rear and 0 the left-facing side. We use the visually inferred convention **relative heading = (270 − filename azimuth) modulo 360**, compatible with Sim2 front=0. Elevation labels follow the filenames and rendered views. The first filename parameter, 10, has unconfirmed distance units and is not used to determine observation pixels.

[Crop audit: 0°](crop-review-az0.jpg) · [90°](crop-review-az90.jpg) · [180°](crop-review-az180.jpg) · [270°](crop-review-az270.jpg) · [Bounds and source hashes](crop_index.json)

## Appearance results

The pinned `openai/clip-vit-base-patch16` revision `{evaluation['model']['revision']}` encodes full frames and padded crops independently on CPU, without dropout or induced noise. Similarities use all 512 dimensions. The historical CLIP resize/center-crop preprocessing is retained; for elongated image crops it can focus on the middle of the body rather than the complete silhouette. Gallery azimuths are 0,30,…330; held-out queries are 15,45,…345, at every elevation and for all series. Source and crop hashes are disjoint between query and gallery.

**Full-frame accuracy: {native['raw']['accuracy']*100:.2f}%. Cropped-frame accuracy: {native['cropped']['accuracy']*100:.2f}%.** These classify the 13 rendered model/paint series; they do not validate identity discrimination between independent same-color vehicles.

![Native and downsampled appearance results](study.png)

The resolution sweep uses 91 queries (one azimuth per series/elevation) and 91 separate reference views offset by 15°. LANCZOS reduces the padded crop width while retaining aspect ratio, followed by pinned CLIP preprocessing. Matched references are independently downsampled and encoded at that width. This is a controlled, narrower protocol than the full native held-out-angle test.

| Query crop width | Original crop references | Matched-resolution references |
|---|---:|---:|
{table}

## Shared embedding explorer

One centered PCA basis is fitted to all 1,404 existing Sim2 Jeep features and all 2,184 new cropped scan features. The first three components show {evaluation['projection_variance_fraction']*100:.2f}% of variation. Dataset/series, elevation and display-axis changes do not refit it. The browser only displays precomputed coordinates and bounded preview sheets. It retains three sheets with at most two concurrent image loads. The basis, row associations, models and hashes are in [projection.json](projection.json); numeric results and all predictions are in [evaluation.json](evaluation.json).

## Applying the SECTR-C camera

The camera replay uses the existing WIRE `Camera`, profile resolver and physical cuboid projection against exact retained Composer truth samples. The binding is `truck01=truck/yellow`; the person is explicitly unsupported because this archive has no person asset. For each observer/time, it computes the target-relative viewing ray and selects a discrete scan by angular distance on the sphere. At nadir, viewing-ray azimuth is degenerate; source image orientation is aligned separately to the camera-projected body forward axis. It isotropically fits/downsamples the crop into the projected physical bounds, pads remaining area, and independently encodes the resulting pixels. No upsampling or viewpoint interpolation adds detail.

The replay retained **{len(camera['observations'])} image/embedding observations**. [Actual pixel crops and enlarged displays](camera/camera-crops.png) · [Camera metadata](camera/index.json) · [512D camera features](camera/embeddings.npy). The interactive explorer includes a camera crop selector. Discrete-angle/model aspect differences remain; target dimensions are authored assumptions, not measurements calibrated from this archive. The replay does not publish ROS, change MHT appearance scoring or modify the running camera feed.

## Reproduction and evidence

The source ZIP remains in SECTR-C and is identified by SHA256 `{audit['source_archive_sha256']}`. The public bundle contains previews, crop bounds, embeddings and evaluation evidence. UCLIP's `prepare_vehicle_scans.py`, `build_vehicle_scan_study.py`, `sample_vehicle_scan_camera.py` and `build_vehicle_scan_view.py` reproduce the stages with the recorded offline model cache and source ZIP. The crop-overrides JSON is an exported snapshot of the SECTR-C dataset review. `{evaluation['independent_reencodes']['images']}` independent single-image re-encodes matched cached rows; maximum absolute feature difference was {evaluation['independent_reencodes']['maximum_absolute_error']:.3g}. This is synthetic appearance/camera-replay evidence, not field perception validation.
'''
    holdout=json.loads((a.study/'elevation_holdout.json').read_text())
    assert holdout['source_crop_index_sha256']==digest(a.crops/'index.json')
    for name,expected in holdout['feature_files'].items():
        if digest(a.study/name)!=expected:raise ValueError('Holdout feature matrix changed')
    elevation_table='\n'.join(f"| {raw['query_elevation_deg']:g}° | {raw['accuracy']*100:.1f}% | {cropped['accuracy']*100:.1f}% |" for raw,cropped in zip(holdout['variants']['raw']['folds'],holdout['variants']['cropped']['folds']))
    elevation_report=f"## Transfer to a held-out elevation\n\nEach of the seven elevations is queried against all six other elevations. Every crop is evaluated once (2,184 total queries per framing), with disjoint source/crop hashes. This tests view transfer rather than supplying the same elevation in the reference gallery. Overall full-frame accuracy is {holdout['variants']['raw']['accuracy']*100:.2f}%; cropped-frame accuracy is {holdout['variants']['cropped']['accuracy']*100:.2f}%.\n\n| Held-out elevation | Full frames | Padded crops |\n|---|---:|---:|\n{elevation_table}\n\n[All elevation-fold predictions](elevation_holdout.json)\n\n"
    report=report.replace('## Shared embedding explorer',elevation_report+'## Shared embedding explorer')
    (a.output/'REPORT.md').write_text(report)
    import markdown
    body=markdown.markdown(report,extensions=['tables'])
    (a.output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>MAAS vehicle scans · CLIP appearance study</title><style>body{max-width:1120px;margin:40px auto;padding:0 20px;font:17px/1.65 system-ui;background:#0b101a;color:#edf2fa}a{color:#bfc9ff}img{max-width:100%}table{border-collapse:collapse}td,th{padding:10px 18px;border-bottom:1px solid #435066}code{overflow-wrap:anywhere;font-size:14px}h1,h2{line-height:1.2}</style>'+body+'</html>')
    verification={'schema':'maas.vehicle_scan_view.v1','generator_sha256':digest(__file__),
                  'renderer_sha256':digest(source/'src/uclip/viz/vehicle_scan_orbit.js'),
                  'template_sha256':digest(source/'src/uclip/viz/vehicle_scan_orbit.html'),
                  'rows':len(rows),'series':len(colors),'preview_max_retained_sheets':3,'preview_max_concurrent_loads':2,
                  'files':{str(f.relative_to(a.output)):{'sha256':digest(f),'bytes':f.stat().st_size} for f in a.output.rglob('*') if f.is_file() and f != a.output/'manifest.json'}}
    write(a.output/'manifest.json',verification)
    print(json.dumps({'output':str(a.output),'rows':len(rows),'preview_sheets':len(sheets),'files':len(verification['files'])}))


if __name__=='__main__':main()
