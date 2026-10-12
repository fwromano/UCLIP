#!/usr/bin/env python3
"""Extend the pinned Sim2 study with native/cropped sphere embeddings."""
import argparse,hashlib,io,json,os,sys,time
from pathlib import Path
from zipfile import ZipFile

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/uclip-vehicle-scans-mpl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image
import torch
from transformers import CLIPImageProcessor,CLIPModel
from uclip.studies.vehicle_scans import MODEL,REVISION


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def encode(images,model,processor):
    inputs=processor(images=images,return_tensors='pt',input_data_format='channels_last')
    with torch.inference_mode():
        raw=model.get_image_features(**inputs).float()
        unit=raw/raw.norm(dim=1,keepdim=True)
    if not torch.isfinite(unit).all():raise ValueError('Nonfinite feature')
    return unit.cpu().numpy().astype(np.float32)


def classify(query,gallery,qlabel,glabel,categories,allowed=None):
    similarity=query@gallery.T
    if allowed is not None:similarity=np.where(allowed,similarity,-np.inf)
    scores=np.stack([similarity[:,glabel==i].max(axis=1) for i in range(len(categories))],axis=1)
    if not np.isfinite(scores).any(axis=1).all():raise ValueError('Query has no gallery candidates')
    predicted=scores.argmax(axis=1)
    return {'accuracy':float(np.mean(predicted==qlabel)), 'queries':len(query),
            'per_series':{c:float(np.mean(predicted[qlabel==i]==i)) for i,c in enumerate(categories) if np.any(qlabel==i)},
            'predicted':predicted.tolist(),'truth':qlabel.tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--crops',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sim2-cache',type=Path,required=True);p.add_argument('--model-cache',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--widths',type=int,nargs='+',default=[16,32,64,128])
    a=p.parse_args();started=time.monotonic()
    audit=json.loads((a.crops/'index.json').read_text());rows=audit['images'];index_hash=digest(a.crops/'index.json')
    if a.output.exists() and not a.resume:raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    checkpoint={'crop_index_sha256':index_hash,'archive_sha256':audit['source_archive_sha256'],
                'model':{'name':MODEL,'revision':REVISION},'widths':a.widths,'generator_sha256':digest(__file__)}
    if (a.output/'checkpoint.json').exists() and json.loads((a.output/'checkpoint.json').read_text())!=checkpoint:
        raise ValueError('Resume checkpoint does not match source/config/code')
    write(a.output/'checkpoint.json',checkpoint)
    if digest(audit['source_archive'])!=audit['source_archive_sha256']:raise ValueError('Archive changed')
    for row in rows:
        if digest(a.crops/row['crop_path'])!=row['crop_sha256']:raise ValueError('Crop changed')
    torch.set_num_threads(4);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    processor=CLIPImageProcessor.from_pretrained(MODEL,revision=REVISION,cache_dir=a.model_cache,local_files_only=True)
    model=CLIPModel.from_pretrained(MODEL,revision=REVISION,cache_dir=a.model_cache,local_files_only=True).to('cpu').eval()
    with ZipFile(audit['source_archive']) as archive:
        matrices={}
        for kind in ['raw','cropped']:
            filename=a.output/f'{kind}_embeddings.npy'
            if a.resume and filename.exists():
                matrix=np.load(filename,allow_pickle=False)
                if matrix.shape!=(len(rows),512):raise ValueError('Incomplete checkpoint feature matrix')
            else:
                chunks=[]
                for begin in range(0,len(rows),16):
                    images=[]
                    for row in rows[begin:begin+16]:
                        if kind=='raw':
                            b=archive.read(row['source_member'])
                            if hashlib.sha256(b).hexdigest()!=row['source_sha256']:raise ValueError('Source changed')
                            image=Image.open(io.BytesIO(b)).convert('RGB')
                        else:image=Image.open(a.crops/row['crop_path']).convert('RGB')
                        images.append(image)
                    chunks.append(encode(images,model,processor))
                    if begin%256==0:print(json.dumps({'phase':kind,'images':begin+len(images),'total':len(rows),'elapsed_sec':time.monotonic()-started}),flush=True)
                matrix=np.concatenate(chunks);np.save(filename,matrix,allow_pickle=False)
            np.testing.assert_allclose(np.linalg.norm(matrix,axis=1),1,atol=2e-6)
            matrices[kind]=matrix
    categories=sorted({r['series'] for r in rows});labels=np.array([categories.index(r['series']) for r in rows])
    query=np.array([i for i,r in enumerate(rows) if r['source_azimuth_deg']%30==15])
    query_hashes={rows[i]['source_sha256'] for i in query};query_crops={rows[i]['crop_sha256'] for i in query}
    gallery=np.array([i for i,r in enumerate(rows) if r['source_azimuth_deg']%30==0 and r['source_sha256'] not in query_hashes and r['crop_sha256'] not in query_crops])
    assert not query_hashes & {rows[i]['source_sha256'] for i in gallery}
    native={kind:classify(matrix[query],matrix[gallery],labels[query],labels[gallery],categories) for kind,matrix in matrices.items()}
    print(json.dumps({'phase':'native_classification','raw_accuracy':native['raw']['accuracy'],'crop_accuracy':native['cropped']['accuracy']}),flush=True)
    rq=np.array([i for i,r in enumerate(rows) if r['source_azimuth_deg']==0])
    rg=np.array([i for i,r in enumerate(rows) if r['source_azimuth_deg']==15])
    assert not {rows[i]['source_sha256'] for i in rq}&{rows[i]['source_sha256'] for i in rg}
    assert not {rows[i]['crop_sha256'] for i in rq}&{rows[i]['crop_sha256'] for i in rg}
    sweep=[];resolution_matrices=[]
    for width in a.widths:
        if width<1:raise ValueError('Widths must be positive')
        encoded=[]
        for begin in range(0,len(rq)+len(rg),16):
            images=[]
            for i in np.concatenate((rq,rg))[begin:begin+16]:
                image=Image.open(a.crops/rows[i]['crop_path']).convert('RGB')
                image=image.resize((width,max(1,round(image.height*width/image.width))),Image.Resampling.LANCZOS)
                images.append(image)
            encoded.append(encode(images,model,processor))
        vectors=np.concatenate(encoded);resolution_matrices.append(vectors)
        native_gallery=classify(vectors[:len(rq)],matrices['cropped'][rg],labels[rq],labels[rg],categories)
        matched=classify(vectors[:len(rq)],vectors[len(rq):],labels[rq],labels[rg],categories)
        sweep.append({'crop_width_px':width,'native_gallery':native_gallery,'matched_resolution_gallery':matched,
                      'mean_cosine_to_native':float(np.mean(np.sum(vectors[:len(rq)]*matrices['cropped'][rq],axis=1)))})
        print(json.dumps({'phase':'resolution','width':width,'native':native_gallery['accuracy'],'matched':matched['accuracy']}),flush=True)
    np.save(a.output/'resolution_embeddings.npy',np.stack(resolution_matrices),allow_pickle=False)
    # Fit a single centered basis for the existing Sim2 Jeep set and all new crops.
    manifest=json.loads((a.sim2_cache/'manifest.json').read_text())
    if manifest['model']!=checkpoint['model']:raise ValueError('Sim2 model/revision mismatch')
    if digest(a.sim2_cache/'embeddings.npy')!=manifest['artifacts']['embeddings.npy']['sha256']:raise ValueError('Sim2 cache changed')
    old_rows=json.loads((a.sim2_cache/'index.json').read_text())['images']
    selected=[r for r in old_rows if r['vehicle_type']=='jeep']
    old=np.load(a.sim2_cache/'embeddings.npy',allow_pickle=False)[[r['embedding_row'] for r in selected]]
    matrix=np.concatenate((old,matrices['cropped'])).astype(np.float64)
    mean=matrix.mean(axis=0);_,singular,vt=np.linalg.svd(matrix-mean,full_matrices=False)
    components=vt[:3];ratio=singular[:3]**2/np.sum(singular**2);coordinates=(matrix-mean)@components.T
    np.testing.assert_allclose(components@components.T,np.eye(3),atol=1e-12)
    projection={'schema':'maas.vehicle_scan_pca.v1','fit':'shared centered SVD over normalized Sim2 Jeep and new cropped scan features',
                'model':checkpoint['model'],'mean':mean.tolist(),'components':components.tolist(),
                'explained_variance_ratio':ratio.tolist(),
                'sim2':[dict(r,coords=coordinates[i].tolist()) for i,r in enumerate(selected)],
                'scans':[dict(r,coords=coordinates[len(selected)+i].tolist()) for i,r in enumerate(rows)]}
    write(a.output/'projection.json',projection)
    # Independent single-image re-encodes verify row/feature correspondence.
    maximum=0.
    for series in categories:
        row=next(r for r in rows if r['series']==series and r['source_azimuth_deg']==0 and r['elevation_deg']==54)
        image=Image.open(a.crops/row['crop_path']).convert('RGB')
        recomputed=encode([image],model,processor)[0];expected=matrices['cropped'][row['embedding_row']]
        maximum=max(maximum,float(np.max(np.abs(recomputed-expected))))
        np.testing.assert_allclose(recomputed,expected,atol=2e-6,rtol=1e-5)
    result={'schema':'maas.vehicle_scan_study.v1','source_crop_index_sha256':index_hash,'source_archive_sha256':audit['source_archive_sha256'],
            'rows':len(rows),'categories':categories,'model':checkpoint['model'],'processor':processor.to_dict(),
            'generation':{'device':'cpu','threads':4,'batch_size':16,'stochastic_noise':False,'mc_dropout':False},
            'native_held_out_azimuth':native,'native_query_rows':query.tolist(),'native_gallery_rows':gallery.tolist(),
            'resolution':sweep,'resolution_query_rows':rq.tolist(),'resolution_gallery_rows':rg.tolist(),
            'projection_variance_fraction':float(ratio.sum()),'independent_reencodes':{'images':len(categories),'maximum_absolute_error':maximum},
            'generator_sha256':digest(__file__),'elapsed_sec':time.monotonic()-started,
            'limits':['One rendered asset per folder; categories distinguish model/paint, not independent same-color vehicles.',
                      'Angle convention visually inferred; source orbit-parameter units unconfirmed.',
                      'Discrete views, crop proposals and ideal resampling; no calibrated optics or image detector likelihood.',
                      'Resolution protocol uses 91 queries and 91 distinct references, one azimuth per series/elevation.']}
    result['artifacts']={name:{'sha256':digest(a.output/name),'bytes':(a.output/name).stat().st_size} for name in ['raw_embeddings.npy','cropped_embeddings.npy','resolution_embeddings.npy','projection.json']}
    write(a.output/'evaluation.json',result)
    print(json.dumps({'phase':'complete','elapsed_sec':result['elapsed_sec'],'output':str(a.output)}),flush=True)


if __name__=='__main__':main()
