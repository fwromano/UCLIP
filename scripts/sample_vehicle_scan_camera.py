#!/usr/bin/env python3
"""Generate crop/feature evidence from exact Composer truth samples and WIRE cameras.

This is an offline camera replay. It does not publish ROS, invent detector
results or add a tracker likelihood. Ground truth IDs stay in this sidecar.
"""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from pyproj import Proj,Transformer
from build_vehicle_scan_study import encode,digest,write,MODEL,REVISION
from uclip.studies.vehicle_scans import closest_view,camera_view_request,crop_rotation,fit_observation_crop


def exact_state(trajectory,requested):
    state=min(trajectory['samples'],key=lambda s:abs(s['sceneTimeSec']-requested))
    if abs(state['sceneTimeSec']-requested)>1e-7:raise ValueError('Replay requires an exact retained truth sample at '+str(requested))
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--crops',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--truth',type=Path,required=True);p.add_argument('--sensors',type=Path,required=True)
    p.add_argument('--wire-root',type=Path,required=True);p.add_argument('--model-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target-map',action='append',required=True,help='truth-id=vehicle/color, e.g. truck01=truck/yellow')
    p.add_argument('--times',type=float,nargs='+',default=[0,5,10,15,20,25,30])
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sys.path.insert(0,str(a.wire_root))
    from simulation.camera import Camera
    from simulation.profiles import runtime_profile
    from simulation.view import projected_target_bounds
    import torch
    from transformers import CLIPImageProcessor,CLIPModel
    config=runtime_profile(json.loads(a.sensors.read_text()));truth=json.loads(a.truth.read_text())
    if config.get('induced_stochastics_enabled',True):
        # Replay intentionally models only ideal geometry/resampling, with its
        # own explicit zero sigmas; it does not silently alter authored settings.
        raise ValueError('Set induced_stochastics_enabled=false for deterministic camera replay')
    trajectories={t['sourceInstanceId']:t for t in truth['trajectories']}
    bindings=dict(value.split('=',1) for value in a.target_map)
    audit=json.loads((a.crops/'index.json').read_text());rows=audit['images']
    evaluation=json.loads((a.study/'evaluation.json').read_text())
    if evaluation['source_crop_index_sha256']!=digest(a.crops/'index.json'):raise ValueError('Study/crops mismatch')
    for r in rows:
        if digest(a.crops/r['crop_path'])!=r['crop_sha256']:raise ValueError('Crop changed')
    native=np.load(a.study/'cropped_embeddings.npy',allow_pickle=False)
    if digest(a.study/'cropped_embeddings.npy')!=evaluation['artifacts']['cropped_embeddings.npy']['sha256']:raise ValueError('Embedding cache changed')
    projection=json.loads((a.study/'projection.json').read_text())
    mean=np.array(projection['mean']);components=np.array(projection['components'])
    geographic=Proj(config['world_crs']);transform=Transformer.from_crs('EPSG:4326',config['world_crs'],always_xy=True)
    def position(state):return np.array([*transform.transform(state['longitude'],state['latitude']),state['altitudeM']])
    def heading(state):return state['headingDeg']-geographic.get_factors(state['longitude'],state['latitude']).meridian_convergence
    a.output.mkdir(parents=True);(a.output/'crops').mkdir();observations=[];images=[];excluded=[]
    for source in config['sources']:
        camera=Camera(**source['camera'])
        for time in a.times:
            observer=exact_state(trajectories[source['instance_id']],time)
            origin=camera.sensor_origin(position(observer),heading(observer))
            for target in config['targets']:
                identity=target['instance_id']
                if identity not in bindings:
                    excluded.append({'observer':source['id'],'scene_time_sec':time,'truth_id':identity,'reason':'No dataset asset binding'})
                    continue
                state=exact_state(trajectories[identity],time)
                observed=camera.observe(position(state),position(observer),heading(observer),target['width_m'],config['ground_hae_m'],np.random.default_rng(0))
                if observed is None:
                    excluded.append({'observer':source['id'],'scene_time_sec':time,'truth_id':identity,'reason':'Outside configured visibility/range/resolution gate'})
                    continue
                dimensions=dict(trajectories[identity]['agentProperties']['dimensionsM']);dimensions['width']=target['width_m']
                bounds=projected_target_bounds(camera,origin,heading(observer),position(state),heading(state),dimensions)
                if bounds is None:raise ValueError('Physical image bounds are unknown')
                if bounds[0]<=0 or bounds[1]<=0 or bounds[2]>=camera.width_px or bounds[3]>=camera.height_px:
                    excluded.append({'observer':source['id'],'scene_time_sec':time,'truth_id':identity,
                                     'reason':'Clipped targets require a partial-body crop adapter'})
                    continue
                request=camera_view_request(camera,origin,heading(observer),position(state),heading(state),dimensions['height'])
                chosen,error=closest_view(rows,bindings[identity],request['relative_heading_deg'],request['elevation_deg'])
                crop=Image.open(a.crops/chosen['crop_path']).convert('RGB')
                image,resampling=fit_observation_crop(crop,[bounds[2]-bounds[0],bounds[3]-bounds[1]],crop_rotation(chosen,request))
                row={'observer':source['id'],'truth_id':identity,'scene_time_sec':time,'series':bindings[identity],
                     'camera':source['camera'],'target_dimensions_m':dimensions,'bbox_px':bounds,'request':request,
                     'selected_embedding_row':chosen['embedding_row'],'selected_source_member':chosen['source_member'],
                     'selected_source_sha256':chosen['source_sha256'],'selected_crop_sha256':chosen['crop_sha256'],
                     'selected_elevation_deg':chosen['elevation_deg'],'selected_relative_heading_deg':chosen['relative_heading_deg'],
                     'viewing_ray_error_deg':error,'resampling':resampling,'crop_path':f'crops/{len(images):03d}.png'}
                image.save(a.output/row['crop_path']);row['image_sha256']=digest(a.output/row['crop_path'])
                images.append(image);observations.append(row)
    if not images:raise ValueError('No supported observations')
    torch.set_num_threads(4);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    processor=CLIPImageProcessor.from_pretrained(MODEL,revision=REVISION,cache_dir=a.model_cache,local_files_only=True)
    model=CLIPModel.from_pretrained(MODEL,revision=REVISION,cache_dir=a.model_cache,local_files_only=True).to('cpu').eval()
    vectors=np.concatenate([encode(images[begin:begin+16],model,processor) for begin in range(0,len(images),16)])
    np.save(a.output/'embeddings.npy',vectors,allow_pickle=False)
    for i,row in enumerate(observations):
        row['embedding_row']=i;row['cosine_to_native_view']=float(vectors[i]@native[row['selected_embedding_row']])
        row['coords']=((vectors[i]-mean)@components.T).tolist()
    result={'schema':'maas.vehicle_scan_camera_replay.v1','model':{'name':MODEL,'revision':REVISION},
            'truth_sha256':digest(a.truth),'sensors_sha256':digest(a.sensors),'crop_index_sha256':digest(a.crops/'index.json'),
            'camera_source_sha256':{f:digest(a.wire_root/'simulation'/f) for f in ['camera.py','profiles.py','view.py']},
            'target_bindings':bindings,'times':a.times,'observations':observations,'excluded':excluded,
            'sampling':'exact retained Composer samples; ideal camera gate and physical cuboid bounds',
            'limits':['Nearest measured viewing ray; no interpolated viewpoint or generated pixels.','In-plane orientation aligned to camera-projected body forward axis.',
                      'Physical dimensions are authored assumptions, not dimensions calibrated from this scan archive.',
                      'Padded isotropic fit; discrete-angle/model aspect differences remain.','No induced stochastics, ROS publishing or MHT appearance scoring.'],
            'embedding_sha256':digest(a.output/'embeddings.npy')}
    write(a.output/'index.json',result)
    sheet=Image.new('RGB',(420,170*len(observations)),'#102033');draw=ImageDraw.Draw(sheet)
    for i,(image,row) in enumerate(zip(images,observations)):
        draw.text((8,i*170+4),f"{row['observer']} {row['truth_id']} t={row['scene_time_sec']:g}s | {image.width} x {image.height} px",fill='white')
        draw.text((8,i*170+20),f"e={row['request']['elevation_deg']:.1f} -> {row['selected_elevation_deg']:g}; ray error {row['viewing_ray_error_deg']:.1f} deg",fill='white')
        sheet.paste(image,(16,i*170+44))
        enlarged=image.resize((image.width*3,image.height*3),Image.Resampling.NEAREST)
        enlarged.thumbnail((300,120),Image.Resampling.NEAREST);sheet.paste(enlarged,(110,i*170+44))
    sheet.save(a.output/'camera-crops.png')
    print(json.dumps({'output':str(a.output),'observations':len(observations),'unsupported':len(excluded)}))


if __name__=='__main__':main()
