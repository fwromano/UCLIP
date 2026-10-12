"""MAAS sphere scan indexing, crop proposals and viewpoint matching.

Source labels remain distinct from detector labels. The 10 filename parameter
has no confirmed units; image scale is determined from the observation camera.
"""
import math
from pathlib import PurePosixPath

import numpy as np
from PIL import Image

MODEL = 'openai/clip-vit-base-patch16'
REVISION = '57c216476eefef5ab752ec549e440a49ae4ae5f3'


def scan_record(name):
    path = PurePosixPath(name)
    if len(path.parts) != 4 or path.suffix.lower() != '.png' or '..' in path.parts:
        raise ValueError('Expected dataset/vehicle/color/parameter_elevation_azimuth.png')
    parameter, elevation, azimuth = map(float, path.stem.split('_'))
    if not all(math.isfinite(v) for v in (parameter, elevation, azimuth)) or not -90 <= elevation <= 90 or not 0 <= azimuth < 360:
        raise ValueError('Invalid scan viewpoint')
    vehicle, color = path.parts[1:3]
    return {'source_member': name, 'vehicle_type': vehicle, 'color': color,
            'series': f'{vehicle}/{color}', 'orbit_parameter': parameter,
            'elevation_deg': elevation, 'source_azimuth_deg': azimuth,
            'relative_heading_deg': (270-azimuth) % 360,
            'heading_status': 'cardinal convention inferred from rendered front/rear/side views'}


def central_yolo_boxes(result, width, height):
    proposals = []
    for box in result.boxes:
        left, top, right, bottom = map(float, box.xyxy[0].tolist())
        fraction = (right-left)*(bottom-top)/(width*height)
        if left <= width/2 <= right and top <= height/2 <= bottom and .003 <= fraction <= .45:
            proposals.append({'bounds': [left,top,right,bottom], 'confidence': float(box.conf[0]),
                              'detector_class': result.names[int(box.cls[0])]})
    return proposals


def mask_bounds(mask):
    y, x = np.nonzero(mask)
    if len(x) < 30:
        return None
    return [int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]


def color_variant_bounds(images):
    arrays = np.stack([np.asarray(image, dtype=np.int16) for image in images])
    mask = np.max(np.ptp(arrays, axis=0), axis=2) > 24
    # Constant background and grid are shared by truck paint variants. Restrict
    # to the known central object region to exclude peripheral lighting changes.
    h,w = mask.shape
    roi = np.zeros_like(mask); roi[int(h*.25):int(h*.82),int(w*.12):int(w*.88)] = True
    return mask_bounds(mask & roi)


def green_body_bounds(image):
    import cv2
    rgb = np.asarray(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h,w = hsv.shape[:2]
    mask = ((hsv[:,:,0] >= 30) & (hsv[:,:,0] <= 90) & (hsv[:,:,1] >= 45) & (hsv[:,:,2] >= 25)).astype(np.uint8)
    roi = np.zeros_like(mask); roi[int(h*.30):int(h*.76),int(w*.24):int(w*.76)] = 1
    mask = cv2.morphologyEx(mask*roi,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    count,labels,stats,centers = cv2.connectedComponentsWithStats(mask)
    choices=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA] >= 30 and
             abs(centers[i][0]-w/2) < w*.17 and abs(centers[i][1]-h/2) < h*.19]
    if not choices: return None
    largest=max(choices,key=lambda i:stats[i,cv2.CC_STAT_AREA])
    base=stats[largest]
    # Retain nearby disconnected paint panels, avoiding isolated horizon grass.
    included=[i for i in choices if abs(centers[i][0]-centers[largest][0]) < max(base[2],40) and
              abs(centers[i][1]-centers[largest][1]) < max(base[3],40)]
    return mask_bounds(np.isin(labels,included))


def padded_union(bounds, size, padding_fraction=.045, minimum_padding=10):
    valid=[b for b in bounds if b is not None]
    if not valid: raise ValueError('No auditable crop proposal; supply a reviewed crop override')
    left=min(b[0] for b in valid);top=min(b[1] for b in valid)
    right=max(b[2] for b in valid);bottom=max(b[3] for b in valid)
    pad=max(minimum_padding,math.ceil(max(right-left,bottom-top)*padding_fraction))
    width,height=size
    return [max(0,math.floor(left-pad)),max(0,math.floor(top-pad)),
            min(width,math.ceil(right+pad)),min(height,math.ceil(bottom+pad))]


def view_direction(relative_heading_deg, elevation_deg):
    a,e=map(math.radians,(relative_heading_deg,elevation_deg))
    return np.array([math.cos(e)*math.sin(a),math.cos(e)*math.cos(a),math.sin(e)])


def closest_view(rows, series, heading_deg, elevation_deg, excluded_hashes=()):
    direction=view_direction(heading_deg,elevation_deg)
    candidates=[r for r in rows if r['series']==series and r['source_sha256'] not in excluded_hashes]
    if not candidates: raise ValueError('No independent view for '+series)
    # At the pole all azimuths describe the same viewing ray. Resolve ties by
    # recorded row; image rotation is matched separately to the camera axes.
    chosen=min(candidates,key=lambda r:(round(1-float(direction@view_direction(r['relative_heading_deg'],r['elevation_deg'])),12),r['embedding_row']))
    error=math.degrees(math.acos(float(np.clip(direction@view_direction(chosen['relative_heading_deg'],chosen['elevation_deg']),-1,1))))
    return chosen,error


def fit_observation_crop(image, bounds_size, rotation_deg=0):
    """Isotropic fit into a pixel-sized observation crop; no fabricated detail."""
    width,height=(max(1,int(math.ceil(v))) for v in bounds_size)
    rotated=image.convert('RGB').rotate(rotation_deg,resample=Image.Resampling.BICUBIC,expand=True,fillcolor=(32,40,45))
    factor=min(width/rotated.width,height/rotated.height,1.0)
    shape=(max(1,round(rotated.width*factor)),max(1,round(rotated.height*factor)))
    resized=rotated.resize(shape,Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(width,height),(32,40,45))
    canvas.paste(resized,((width-resized.width)//2,(height-resized.height)//2))
    return canvas,{'isotropic_scale':factor,'content_size_px':list(shape),'output_size_px':[width,height],
                   'upsampled':False,'rotation_deg':rotation_deg}


def camera_view_request(camera, origin, camera_heading, target, target_heading, target_height):
    """Viewing ray and forward-axis image direction in the camera's metric frame."""
    center=np.asarray(target,dtype=float)+[0,0,target_height/2]
    offset=np.asarray(origin,dtype=float)-center
    horizontal=float(np.hypot(offset[0],offset[1]))
    heading=(math.degrees(math.atan2(offset[0],offset[1]))-target_heading)%360 if horizontal>1e-8 else 0.
    elevation=math.degrees(math.atan2(offset[2],horizontal))
    rotation=camera.rotation(camera_heading)
    forward=np.array([math.sin(math.radians(target_heading)),math.cos(math.radians(target_heading)),0.])
    def project(point):
        optical=rotation.T@(point-np.asarray(origin))
        if optical[2]<=1e-6:raise ValueError('Target is behind camera')
        return (camera.K@optical)[:2]/optical[2]
    direction=project(center+forward)-project(center)
    return {'relative_heading_deg':heading,'elevation_deg':elevation,'range_m':float(np.linalg.norm(offset)),
            'forward_image_direction':direction.tolist(),'azimuth_defined':horizontal>1e-8}


def crop_rotation(source_row, requested):
    angle,elevation=map(math.radians,(source_row['relative_heading_deg'],source_row['elevation_deg']))
    source=np.array([math.sin(angle),math.sin(elevation)*math.cos(angle)])
    desired=np.array(requested['forward_image_direction'])
    if np.linalg.norm(source)<1e-6 or np.linalg.norm(desired)<1e-6:return 0.
    difference=math.degrees(math.atan2(desired[1],desired[0])-math.atan2(source[1],source[0]))
    # PIL positive angles are counterclockwise; screen Y increases downwards.
    return (-difference+180)%360-180
