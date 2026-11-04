import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


path=Path(__file__).resolve().parents[1]/'tools/prepare_soccernet_config.py'
spec=importlib.util.spec_from_file_location('prepare_soccernet_config',path)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


class SoccerNetConfigTests(unittest.TestCase):
    def source(self,root):
        for number,game in enumerate((4,4,5,6)):
            folder=root/'train'/f'SNMOT-{number:03d}';(folder/'img1').mkdir(parents=True);(folder/'gt').mkdir()
            sid=folder.name
            (folder/'seqinfo.ini').write_text(f'[Sequence]\nname={sid}\nimDir=img1\nframeRate=25\nseqLength=750\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n')
            (folder/'gameinfo.ini').write_text(f'[Sequence]\nname={sid}\ngameID={game}\nnum_tracklets=4\ntrackletID_1= player team left;10\ntrackletID_2= goalkeepers team left;y\ntrackletID_3= referee;main\ntrackletID_4= ball;1\n')
            (folder/'gt/gt.txt').write_text('1,1,914,855,55,172,1,-1,-1,-1\n1,4,5,5,3,3,1,-1,-1,-1\n')

    def test_real_metadata_shape_keeps_games_and_review_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.source(root)
            config,report=helper.prepare_config(root,root/'config.json')
            first=config['sequences'][0]
            self.assertEqual(first['track_classes']['1'],'player team left')
            self.assertIn('goalkeepers team left',first['person_track_classes'])
            self.assertEqual(first['ignored_track_classes'],['ball'])
            self.assertEqual(first['split'],config['sequences'][1]['split'])
            self.assertEqual({s['split'] for s in config['sequences']},{'train','val','test'})
            self.assertEqual(report['games'],3)
            self.assertFalse(config['annotations_reviewed'])
            self.assertFalse(config['person_labels_complete'])
            self.assertFalse(config['training_eligible'])
            self.assertEqual(first['frame_stride'],25)

    def test_unknown_class_is_not_silently_ignored(self):
        for label in ('unknown role','other object'):
            with self.subTest(label=label),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);self.source(root)
                p=root/'train/SNMOT-000/gameinfo.ini';p.write_text(p.read_text().replace('player team left',label))
                with self.assertRaises(ValueError):helper.prepare_config(root,root/'config.json')

    def test_documented_other_humans_are_person_not_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.source(root)
            p=root/'train/SNMOT-000/gameinfo.ini'
            p.write_text(p.read_text().replace('player team left;10','other;medical staff'))
            config,_=helper.prepare_config(root,root/'config.json')
            sequence=config['sequences'][0]
            self.assertEqual(sequence['track_classes']['1'],'other')
            self.assertIn('other',sequence['person_track_classes'])
            self.assertNotIn('other',sequence['ignored_track_classes'])
            self.assertIn('ball',sequence['ignored_track_classes'])
            self.assertFalse(config['annotations_reviewed'])

    def test_missing_track_mapping_and_overwrite_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.source(root)
            target=root/'config.json';helper.prepare_config(root,target)
            with self.assertRaises(FileExistsError):helper.prepare_config(root,target)
            p=root/'train/SNMOT-000/gt/gt.txt';p.write_text('1,999,1,1,3,3,1,-1,-1,-1\n')
            with self.assertRaises(ValueError):helper.prepare_config(root,root/'other.json')

    def test_default_origin_preserves_edge_boxes_through_staging_and_etl(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow data extra required')
        from touchline.data.soccernet import stage_soccernet
        from touchline.data.etl import build_release
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.source(root)
            for index,folder in enumerate(sorted((root/'train').iterdir())):
                meta=folder/'seqinfo.ini'
                meta.write_text(meta.read_text().replace('imWidth=1920','imWidth=64').replace('imHeight=1080','imHeight=64'))
                Image.new('RGB',(64,64),(index*50,20,30)).save(folder/'img1/000001.jpg')
                (folder/'gt/gt.txt').write_text('1,1,0,5,20,10,1,-1,-1,-1\n1,4,5,5,3,3,1,-1,-1,-1\n')
            config,_=helper.prepare_config(root,root/'config.json')
            self.assertTrue(all(s['bbox_origin']==0 for s in config['sequences']))
            config.update(annotations_reviewed=True,person_labels_complete=True,training_eligible=True,reviewer_id='synthetic-test')
            (root/'config.json').write_text(json.dumps(config))
            staged=stage_soccernet(root/'config.json',root/'staged')
            # Generated images exercise conversion only and must never train a model.
            etl_config=json.loads(staged.read_text());etl_config.update(fixture_only=True,training_eligible=False)
            staged.write_text(json.dumps(etl_config))
            release=build_release(staged,root/'release')
            self.assertTrue(release['fixture_only'])
            for sample in release['samples']:
                self.assertEqual(sample['objects'][0]['bbox_xyxy'],[0,5,20,15])
