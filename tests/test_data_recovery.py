import hashlib,io,tempfile,unittest
from pathlib import Path
from PIL import Image
from touchline.data.recovery import restore_image


class RestoreTests(unittest.TestCase):
    def setUp(self):
        b=io.BytesIO();Image.new('RGB',(16,16),'green').save(b,format='JPEG');self.raw=b.getvalue()
        n=io.BytesIO()
        with Image.open(io.BytesIO(self.raw)) as image:image.convert('RGB').save(n,format='PNG')
        self.sample={'source_sha256':hashlib.sha256(self.raw).hexdigest(),'image_sha256':hashlib.sha256(n.getvalue()).hexdigest(),'image_path':'images/'+'a'*24+'.png'}
    def test_exact_original_and_normalized_hashes_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertGreater(restore_image(self.raw,self.sample,tmp),0)
            self.assertTrue((Path(tmp)/self.sample['image_path']).is_file())
    def test_mismatched_original_or_normalized_image_is_not_rehashed(self):
        for change in ({'source_sha256':'0'*64},{'image_sha256':'0'*64}):
            with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):restore_image(self.raw,{**self.sample,**change},tmp)
    def test_unsafe_image_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):restore_image(self.raw,{**self.sample,'image_path':'../escape.png'},tmp)
