import json
from pathlib import Path
import tempfile
import unittest
from touchline.observation_pages import observation_page


class ObservationPageTests(unittest.TestCase):
    def test_pages_preserve_all_ids_and_end_without_duplicate_or_missing_row(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'observations.jsonl'
            path.write_text('\n'.join(json.dumps({'id':i}) for i in range(5)),encoding='utf-8')
            result=[];offset=0
            while offset is not None:
                page=observation_page(path,offset=offset,limit=2)
                result.extend(page['observations']);offset=page['next_offset']
            self.assertEqual([r['id'] for r in result],list(range(5)))

    def test_invalid_page_and_noncanonical_file_are_rejected(self):
        for offset,limit in [(-1,2),(0,5001),(False,2),(0,0)]:
            with self.subTest(offset=offset,limit=limit), self.assertRaises(ValueError):
                observation_page('observations.jsonl',offset=offset,limit=limit)
        with self.assertRaises(ValueError):observation_page('.env')
