import json
from pathlib import Path
import tempfile
import unittest
try:
    from PIL import Image
except ImportError:
    Image=None
from touchline.data.soccernet import stage_soccernet
from touchline.data.etl import build_release


@unittest.skipIf(Image is None,'Pillow data extra required')
class SoccerNetTests(unittest.TestCase):
    def source(self,root):
        sequences=[]
        for i,split in enumerate(('train','val','test')):
            folder=root/f'seq{i}';(folder/'img1').mkdir(parents=True)
            Image.new('RGB',(64,64),(i*60,10,20)).save(folder/'img1/000001.jpg')
            (folder/'gt.txt').write_text('1,1,10,12,20,30,1,-1,-1,-1\n1,2,1,1,3,3,1,-1,-1,-1\n')
            sequences.append({'id':f'seq{i}','capture_group_id':f'match{i}','split':split,
                'images_root':str(folder/'img1'),'annotations':str(folder/'gt.txt'),'frame_stride':1,'bbox_origin':0,
                'track_classes':{'1':'player','2':'ball'},'person_track_classes':['player'],'ignored_track_classes':['ball']})
        config={'release_id':'real-format-fixture','dataset_version':'1','reviewer_id':'test-only',
            'annotations_reviewed':True,'person_labels_complete':True,'training_eligible':True,'sequences':sequences}
        path=root/'config.json';path.write_text(json.dumps(config));return path,config

    def test_conversion_filters_ball_and_preserves_explicit_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,config=self.source(root)
            staged=stage_soccernet(path,root/'staged')
            # This fixture only exercises conversion. Do not train on generated images.
            generated=json.loads(staged.read_text());generated.update(fixture_only=True,training_eligible=False)
            staged.write_text(json.dumps(generated))
            release=build_release(staged,root/'release')
            self.assertTrue(release['fixture_only'])
            for sample in release['samples']:
                self.assertEqual(len(sample['objects']),1)
                self.assertEqual(sample['objects'][0]['bbox_xyxy'],[10,12,30,42])
                self.assertEqual(sample['split'],config['sequences'][int(sample['source_image_ref'][3])]['split'])

    def test_unknown_track_and_detection_rows_fail(self):
        for case in ('unknown','detection'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);path,config=self.source(root)
                if case=='unknown':config['sequences'][0]['track_classes'].pop('1')
                else:(root/'seq0/gt.txt').write_text('1,1,10,12,20,30,0.7,-1,-1,-1\n')
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):stage_soccernet(path,root/'staged')

    def test_same_match_across_splits_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,config=self.source(root)
            config['sequences'][1]['capture_group_id']='match0'
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):stage_soccernet(path,root/'staged')

    def test_explicit_one_based_pixel_origin_converts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,config=self.source(root)
            for item in config['sequences']:item['bbox_origin']=1
            path.write_text(json.dumps(config));stage_soccernet(path,root/'staged')
            coco=json.loads((root/'staged/annotations-000.json').read_text())
            self.assertEqual(coco['annotations'][0]['bbox'],[9,11,20,30])

    def test_invalid_box_on_unselected_frame_does_not_block_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,config=self.source(root)
            for sequence in config['sequences']:sequence['frame_stride']=25
            with (root/'seq0/gt.txt').open('a') as handle:
                handle.write('403,1,1330,542,0,82,1,-1,-1,-1\n')
            path.write_text(json.dumps(config));stage_soccernet(path,root/'staged')
            coco=json.loads((root/'staged/annotations-000.json').read_text())
            self.assertEqual([i['id'] for i in coco['images']],[1])
            self.assertEqual(len(coco['annotations']),1)
            report=json.loads((root/'staged/staging-report.json').read_text())['sequences'][0]
            self.assertEqual(report['unselected_invalid_box_rows'],1)
            self.assertEqual(report['unselected_invalid_box_examples'][0]['frame'],403)

    def test_invalid_box_on_selected_frame_still_fails_with_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,config=self.source(root)
            for sequence in config['sequences']:sequence['frame_stride']=25
            with (root/'seq0/gt.txt').open('a') as handle:
                handle.write('401,1,1330,542,0,82,1,-1,-1,-1\n')
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,r'seq0, line 3, frame=401.*nonfinite/nonpositive'):
                stage_soccernet(path,root/'staged')

    def test_stride_is_validated_before_reading_rows(self):
        for stride in (0,-1,True):
            with self.subTest(stride=stride),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);path,config=self.source(root)
                config['sequences'][0]['frame_stride']=stride
                path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError,'positive frame_stride'):
                    stage_soccernet(path,root/'staged')


if __name__=='__main__':unittest.main()
