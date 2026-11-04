import copy
import json
from pathlib import Path
import tempfile
import unittest
from touchline.data.manifest import validate_manifest
from touchline.data.inventory import inventory

try:
    from PIL import Image
except ImportError:
    Image=None


def fixture():
    return {'release_id':'fixture','dataset_version':'1','fixture_only':True,'class_map':{'0':'background','1':'person'},
      'split_policy':{'mode':'capture_group'},'samples':[
        {'sample_id':str(i),'source_id':str(i),'source_sha256':str(i+1)*64,'capture_group_id':f'g{i}',
         'source_kind':'still_image','pts':None,'time_base':None,'frame_index':None,'width':32,'height':32,
         'image_sha256':str(i+4)*64,'review_status':'human_reviewed','reviewer_id':'synthetic','training_eligible':False,
         'is_negative':False,'objects':[{'class_id':1,'bbox_xyxy':[1,2,9,10],'annotation_status':'human_reviewed','origin':'manual'}],
         'split':split} for i,split in enumerate(('train','val','test'))]}


class ManifestTests(unittest.TestCase):
    def test_fixture_is_metadata_not_training(self):
        self.assertEqual(validate_manifest(fixture()),[])
        self.assertTrue(validate_manifest(fixture(),training=True))

    def test_group_and_hash_leakage(self):
        for key in ('capture_group_id','image_sha256','source_sha256'):
            data=fixture();data['samples'][1][key]=data['samples'][0][key]
            self.assertTrue(any('leakage' in e for e in validate_manifest(data)))

    def test_unreviewed_and_reversed_box(self):
        data=fixture();data['samples'][0]['objects'][0]['annotation_status']='proposal'
        data['samples'][1]['objects'][0]['bbox_xyxy']=[20,0,1,2]
        self.assertTrue(validate_manifest(data))

    def test_still_images_do_not_invent_video_timestamps(self):
        data=fixture();data['samples'][0]['pts']=0
        self.assertTrue(any('invent' in e for e in validate_manifest(data)))

    def test_inventory_does_not_call_generic_json_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'stats.csv').write_text('player,goals\na,1\n');(p/'events.json').write_text('{"events":[]}')
            result=inventory(p)
            self.assertFalse(result['training_ready']);self.assertEqual(result['extension_counts']['.csv'],1)


@unittest.skipIf(Image is None,'ETL image tests require the optional data extra (Pillow)')
class ETLTests(unittest.TestCase):
    def make_source(self,tmp,format):
        root=Path(tmp);(root/'images').mkdir();(root/'labels').mkdir()
        images=[];annotations=[];groups={}
        for i in range(3):
            name=f'{i}.png';Image.new('RGB',(32,32),(i*70,20,30)).save(root/'images'/name)
            images.append({'id':i,'file_name':name,'width':32,'height':32})
            annotations.append({'id':i,'image_id':i,'category_id':1,'bbox':[8,8,16,16]})
            (root/'labels'/f'{i}.txt').write_text('0 0.5 0.5 0.5 0.5\n')
            groups['input/'+name]=f'capture-{i}'
        (root/'coco.json').write_text(json.dumps({'images':images,'annotations':annotations,'categories':[{'id':1,'name':'player'}]}))
        item={'id':'input','format':format,'images_root':'images','annotations':'coco.json','labels_root':'labels','classes':['player']}
        config={'release_id':'synthetic-etl','dataset_version':'1','reviewer_id':'synthetic','annotations_reviewed':True,
         'person_labels_complete':True,'training_eligible':False,'fixture_only':True,'class_aliases':{'player':1},
         'capture_groups':groups,'inputs':[item]}
        (root/'config.json').write_text(json.dumps(config));return root,config

    def test_coco_and_yolo_transform_to_same_boxes(self):
        from touchline.data.etl import build_release
        for format in ('coco_boxes','yolo_boxes'):
            with self.subTest(format=format),tempfile.TemporaryDirectory() as tmp:
                root,config=self.make_source(tmp,format)
                release=build_release(root/'config.json',root/'release')
                self.assertEqual(len(release['samples']),3)
                self.assertEqual({x['split'] for x in release['samples']},{'train','val','test'})
                self.assertEqual(release['samples'][0]['objects'][0]['bbox_xyxy'],[8,8,24,24])
                self.assertEqual(validate_manifest(release),[])

    def test_missing_yolo_label_does_not_become_negative(self):
        from touchline.data.etl import build_release
        with tempfile.TemporaryDirectory() as tmp:
            root,_=self.make_source(tmp,'yolo_boxes');(root/'labels/0.txt').unlink()
            with self.assertRaises(ValueError):build_release(root/'config.json',root/'release')
            self.assertFalse((root/'release/release.json').exists())
            self.assertEqual(json.loads((root/'release/etl-report.json').read_text())['status'],'failed')

    def test_unknown_class_and_missing_group_fail(self):
        from touchline.data.etl import build_release
        for edit in ('class_aliases','capture_groups'):
            with self.subTest(edit=edit),tempfile.TemporaryDirectory() as tmp:
                root,config=self.make_source(tmp,'coco_boxes');config[edit]={'other':1} if edit=='class_aliases' else {}
                (root/'config.json').write_text(json.dumps(config))
                with self.assertRaises(ValueError):build_release(root/'config.json',root/'release')

    def test_pixel_corner_and_csv_need_explicit_semantics(self):
        from touchline.data.etl import build_release
        for format in ('xyxy_boxes','csv_boxes'):
            with self.subTest(format=format),tempfile.TemporaryDirectory() as tmp:
                root,config=self.make_source(tmp,format)
                item=config['inputs'][0];item['coordinate_units']='pixels'
                for i in range(3):(root/'labels'/f'{i}.txt').write_text('0 8 8 24 24\n')
                (root/'boxes.csv').write_text('file,label,x1,y1,x2,y2\n'+''.join(f'{i}.png,player,8,8,24,24\n' for i in range(3)))
                item['annotations']='boxes.csv';item['columns']={'image':'file','class':'label','x1':'x1','y1':'y1','x2':'x2','y2':'y2'}
                (root/'config.json').write_text(json.dumps(config))
                release=build_release(root/'config.json',root/'release')
                self.assertEqual(release['samples'][0]['objects'][0]['bbox_xyxy'],[8,8,24,24])


if __name__=='__main__':unittest.main()
