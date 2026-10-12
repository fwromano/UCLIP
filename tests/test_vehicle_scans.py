import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image
from uclip.studies.vehicle_scans import scan_record,color_variant_bounds,padded_union,closest_view,fit_observation_crop,camera_view_request,crop_rotation


class VehicleScanTests(unittest.TestCase):
    def test_cardinal_convention_and_filename_validation(self):
        for native,relative in ((0,270),(90,180),(180,90),(270,0)):
            row=scan_record(f'data/truck/yellow/10_54_{native}.png')
            self.assertEqual(row['relative_heading_deg'],relative)
            self.assertEqual(row['elevation_deg'],54)
        for path in ('data/../yellow/10_54_0.png','data/truck/yellow/10_100_0.png','data/truck/yellow/10_54_360.png'):
            with self.assertRaises(ValueError):scan_record(path)

    def test_paint_variants_have_one_shared_crop_and_constant_background_is_excluded(self):
        first=np.full((100,100,3),80,np.uint8);second=first.copy()
        first[40:60,35:65]=[250,20,20];second[40:60,35:65]=[20,250,20]
        bounds=color_variant_bounds([Image.fromarray(first),Image.fromarray(second)])
        self.assertEqual(bounds,[35,40,65,60])
        self.assertEqual(padded_union([bounds],[100,100]),[25,30,75,70])

    def test_spherical_matching_wraps_heading_and_handles_poles_without_fake_interpolation(self):
        rows=[dict(series='truck/yellow',source_sha256=str(i),embedding_row=i,relative_heading_deg=h,elevation_deg=e)
              for i,(h,e) in enumerate(((359,54),(180,90),(0,0)))]
        row,error=closest_view(rows,'truck/yellow',1,54)
        self.assertEqual(row['embedding_row'],0);self.assertLess(error,2)
        row,error=closest_view(rows,'truck/yellow',15,90)
        self.assertEqual(row['embedding_row'],1);self.assertAlmostEqual(error,0)
        row,_=closest_view(rows,'truck/yellow',1,54,excluded_hashes={'0'})
        self.assertNotEqual(row['embedding_row'],0)

    def test_observation_fit_preserves_aspect_and_small_pixel_sizes(self):
        image=Image.new('RGB',(100,50),'white')
        output,metadata=fit_observation_crop(image,[4.2,10.1])
        self.assertEqual(output.size,(5,11));self.assertEqual(metadata['content_size_px'],[5,2])
        self.assertFalse(metadata['upsampled'])
        self.assertEqual(output.getpixel((2,4)),(255,255,255))
        self.assertEqual(output.getpixel((2,0)),(32,40,45))
        output,metadata=fit_observation_crop(image,[10,20],rotation_deg=90)
        self.assertEqual(metadata['content_size_px'],[10,20])
        self.assertEqual(output.size,(10,20))

    def test_nadir_crop_orientation_matches_camera_forward_axis(self):
        class NadirCamera:
            K=np.array([[100,0,320],[0,100,320],[0,0,1.]])
            def rotation(self,heading):return np.diag([1.,-1.,-1.])
        request=camera_view_request(NadirCamera(),[0,0,100],0,[0,0,0],0,2)
        self.assertEqual(request['elevation_deg'],90)
        self.assertFalse(request['azimuth_defined'])
        self.assertLess(request['forward_image_direction'][1],0)
        source={'relative_heading_deg':270,'elevation_deg':90}
        self.assertAlmostEqual(crop_rotation(source,request),-90)
        # A left-facing source with a red nose must point upward after alignment.
        image=Image.new('RGB',(20,10),'white')
        for x in range(3):
            for y in range(10):image.putpixel((x,y),(255,0,0))
        output,_=fit_observation_crop(image,[10,20],crop_rotation(source,request))
        self.assertEqual(output.getpixel((5,1)),(255,0,0))
        self.assertEqual(output.getpixel((5,18)),(255,255,255))


if __name__=='__main__':unittest.main()
