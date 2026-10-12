#!/usr/bin/env python3
"""Evaluate cross-elevation transfer from the completed deterministic scan cache."""
import argparse,json,os,hashlib
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--crops',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    audit=json.loads((a.crops/'index.json').read_text());rows=audit['images']
    checkpoint=json.loads((a.study/'checkpoint.json').read_text())
    if checkpoint['crop_index_sha256']!=digest(a.crops/'index.json'):raise ValueError('Cache/crop association mismatch')
    categories=sorted({r['series'] for r in rows});labels=np.array([categories.index(r['series']) for r in rows])
    elevations=np.array([r['elevation_deg'] for r in rows]);levels=sorted(set(elevations))
    result={'schema':'maas.vehicle_scan_elevation_holdout.v1','categories':categories,'model':checkpoint['model'],
            'protocol':'Each elevation is queried against every other elevation, with source and crop hashes disjoint.',
            'source_crop_index_sha256':digest(a.crops/'index.json'),'variants':{},'feature_files':{}}
    for kind in ['raw','cropped']:
        path=a.study/f'{kind}_embeddings.npy';matrix=np.load(path,allow_pickle=False)
        assert matrix.shape==(len(rows),512)
        np.testing.assert_allclose(np.linalg.norm(matrix,axis=1),1,atol=2e-6)
        similarities=matrix@matrix.T;folds=[];all_correct=[]
        for elevation in levels:
            query=np.flatnonzero(elevations==elevation);source_hashes={rows[i]['source_sha256'] for i in query};crop_hashes={rows[i]['crop_sha256'] for i in query}
            gallery=np.array([i for i in np.flatnonzero(elevations!=elevation) if rows[i]['source_sha256'] not in source_hashes and rows[i]['crop_sha256'] not in crop_hashes])
            scores=np.stack([similarities[np.ix_(query,gallery[labels[gallery]==i])].max(axis=1) for i in range(len(categories))],axis=1)
            predicted=scores.argmax(axis=1);correct=predicted==labels[query];all_correct.extend(correct)
            folds.append({'query_elevation_deg':float(elevation),'queries':len(query),'gallery_rows':len(gallery),'accuracy':float(correct.mean()),
                          'per_series':{c:float(correct[labels[query]==i].mean()) for i,c in enumerate(categories)},'query_rows':query.tolist(),'predicted':predicted.tolist(),'truth':labels[query].tolist()})
        result['variants'][kind]={'accuracy':float(np.mean(all_correct)),'queries':len(all_correct),'folds':folds}
        result['feature_files'][path.name]=digest(path)
    write(a.output,result)
    print(json.dumps({kind:metrics['accuracy'] for kind,metrics in result['variants'].items()}))


if __name__=='__main__':main()
