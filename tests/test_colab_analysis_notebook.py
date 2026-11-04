"""Exercise safe notebook branches without Colab, GPU, Drive or provider access."""
import ast
import datetime
import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import sys
from unittest.mock import Mock,patch


class ColabAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        book=json.loads((root/'notebooks/02_video_analysis.ipynb').read_text(encoding='utf-8'))
        cls.code=[c['source'] for c in book['cells'] if c['cell_type']=='code']

    def analysis_cell(self,enabled):
        source=next(s for s in self.code if s.startswith('RUN_ANALYSIS ='))
        tree=ast.parse(source)
        if enabled:tree.body[0].value=ast.Constant(True)
        return compile(ast.fix_missing_locations(tree),'colab-analysis-cell','exec')

    def test_opening_analysis_cell_does_not_launch_compute(self):
        cli=Mock();path=Mock()
        exec(self.analysis_cell(False),{'cli':cli,'Path':path,'EXECUTION_MODE':'cpu_replay'})
        cli.assert_not_called();path.assert_not_called()

    def test_enabled_smoke_run_uses_chosen_weights_cap_and_sampler(self):
        with tempfile.TemporaryDirectory() as tmp:
            def local_path(value):
                return Path(tmp) if value=='/content/touchline-analysis' else Path(value)
            def fake_cli(*args):
                output=args[args.index('--output')+1];output.mkdir()
                (output/'manifest.json').write_text(json.dumps({'run_id':'fixture-analysis','status':'partial','detector':{'checkpoint':'a'*64}}),encoding='utf-8')
            cli=Mock(side_effect=fake_cli)
            scope={'Path':local_path,'cli':cli,'datetime':datetime,'json':json,'runtime':{},'DEVICE':'cuda','EXECUTION_MODE':'cuda_analysis','CUDA_TRACKING_CHECK_PASSED':True,
                'VIDEO_PATH':Path('/fixture/video.mp4'),'CHECKPOINT_PATH':Path('/fixture/candidate.pt'),
                'checkpoint_digest':lambda p:'a'*64,'CHECKPOINT_SHA256':'a'*64}
            with patch.object(Path,'is_file',return_value=True):exec(self.analysis_cell(True),scope)
            args=cli.call_args.args
            self.assertEqual(args[0],'analyze');self.assertEqual(args[args.index('--checkpoint')+1],scope['CHECKPOINT_PATH'])
            self.assertEqual(args[args.index('--max-selected')+1],240)
            self.assertEqual(args[args.index('--fps')+1],12)
            self.assertEqual(args[args.index('--tracker')+1],'bytetrack-cuda')
            self.assertEqual(args[args.index('--mode')+1],'fixed')
            self.assertIn('--record-proposals',args)
            self.assertNotIn('--allow-weight-download',args);self.assertNotIn('--env-file',args)

    def exporter(self):
        source=next(s for s in self.code if s.startswith('def save_analysis_bundle'))
        function=ast.parse(source).body[0]
        scope={'Path':Path,'datetime':datetime,'json':json,'shutil':shutil}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'colab-export-function','exec'),scope)
        return scope['save_analysis_bundle']

    def test_drive_bundle_verifies_known_outputs_and_excludes_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)/'work';run=work/'run';run.mkdir(parents=True)
            (run/'manifest.json').write_text('{"run_id":"fixture-run","status":"partial"}',encoding='utf-8')
            (work/'private.txt').write_text('not-a-product-artifact',encoding='utf-8')
            preview=work/'preview-fixture';preview.mkdir()
            (preview/'overlay24.mp4').write_bytes(b'fixture-video-transfer')
            (preview/'preview.json').write_text('{}',encoding='utf-8')
            destination=self.exporter()(work,run,Path(tmp)/'drive')
            receipt=json.loads((destination/'TRANSFER_COMPLETE.json').read_text(encoding='utf-8'))
            self.assertEqual(receipt['run_id'],'fixture-run')
            self.assertEqual(set(receipt['sha256']),{'run/manifest.json','preview-fixture/overlay24.mp4','preview-fixture/preview.json'})
            self.assertFalse((destination/'private.txt').exists())

    def test_failed_transfer_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)/'work';run=work/'run';run.mkdir(parents=True);drive=Path(tmp)/'drive'
            (run/'manifest.json').write_text('{"run_id":"fixture-run"}',encoding='utf-8')
            def corrupt_copy(source,target):target.write_bytes(b'corrupted-transfer')
            with patch('shutil.copyfile',side_effect=corrupt_copy):
                with self.assertRaises(AssertionError):self.exporter()(work,run,drive)
            self.assertEqual(list(drive.rglob('TRANSFER_COMPLETE.json')),[])
            self.assertEqual(json.loads((run/'manifest.json').read_text(encoding='utf-8'))['run_id'],'fixture-run')

    def input_scope(self,root,*,execute_selection=False,mode='cuda_analysis'):
        source=next(s for s in self.code if s.startswith('# Colab input cell.'))
        nodes=[];tail=False
        for node in ast.parse(source).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHECKPOINT_OVERRIDE' for t in node.targets):tail=True
            if isinstance(node,ast.FunctionDef) or tail and execute_selection:nodes.append(node)
        scope={'Path':Path,'json':json,'os':os,'hashlib':hashlib,'TOUCHLINE_ROOT':root,'cli':Mock(),'EXECUTION_MODE':mode}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'colab-input-functions','exec'),scope)
        return scope

    def test_cpu_replay_does_not_require_or_hash_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope=self.input_scope(Path(tmp),execute_selection=True,mode='cpu_replay')
            self.assertIsNone(scope['CHECKPOINT_PATH'])
            self.assertIsNone(scope['CHECKPOINT_SHA256'])
            scope['cli'].assert_not_called()

    def test_cpu_runtime_branch_never_imports_torch_or_runs_cuda_check(self):
        source=next(s for s in self.code if s.startswith('EXECUTION_MODE ='))
        source=source.replace("EXECUTION_MODE = 'cpu_analysis'","EXECUTION_MODE = 'cpu_replay'",1)
        import builtins
        original=builtins.__import__
        def safe_import(name,*args,**kwargs):
            if name in ('torch','torchvision'):raise AssertionError('CPU replay must not import Torch')
            return original(name,*args,**kwargs)
        subprocess=Mock();cli=Mock()
        scope={'subprocess':subprocess,'sys':sys,'json':json,'cli':cli,'REPO_DIR':Path('/fixture')}
        with patch('builtins.__import__',side_effect=safe_import):exec(compile(source,'cpu-runtime-cell','exec'),scope)
        self.assertEqual(scope['DEVICE'],'cpu')
        self.assertFalse(scope['CUDA_TRACKING_CHECK_PASSED'])
        self.assertEqual(subprocess.run.call_count,1)  # dependency installation only, mocked
        cli.assert_called_once_with('doctor')

    def test_enabled_cpu_analysis_uses_cpu_tracker_and_six_samples_per_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            def local_path(value):return Path(tmp) if value=='/content/touchline-analysis' else Path(value)
            def fake_cli(*args):
                output=args[args.index('--output')+1];output.mkdir()
                (output/'manifest.json').write_text(json.dumps({'run_id':'cpu-fixture','status':'partial','detector':{'checkpoint':'a'*64}}),encoding='utf-8')
            cli=Mock(side_effect=fake_cli)
            scope={'Path':local_path,'cli':cli,'datetime':datetime,'json':json,'runtime':{},'DEVICE':'cpu',
                'EXECUTION_MODE':'cpu_analysis','CUDA_TRACKING_CHECK_PASSED':False,
                'VIDEO_PATH':Path('/fixture/video.mp4'),'CHECKPOINT_PATH':Path('/fixture/candidate.pt'),
                'checkpoint_digest':lambda p:'a'*64,'CHECKPOINT_SHA256':'a'*64}
            with patch.object(Path,'is_file',return_value=True):exec(self.analysis_cell(True),scope)
            args=cli.call_args.args
            self.assertEqual(args[args.index('--device')+1],'cpu')
            self.assertEqual(args[args.index('--tracker')+1],'bytetrack')
            self.assertEqual(args[args.index('--fps')+1],6)
            self.assertEqual(args[args.index('--max-selected')+1],120)
            self.assertNotIn('--env-file',args)

    def test_default_cpu_runtime_checks_cpu_nms_without_cuda_probe(self):
        source=next(s for s in self.code if s.startswith('EXECUTION_MODE ='))
        subprocess=Mock();cli=Mock()
        scope={'subprocess':subprocess,'sys':sys,'json':json,'cli':cli,'REPO_DIR':Path('/fixture')}
        with patch('torch.cuda.is_available',side_effect=AssertionError('CPU mode must not probe CUDA')):
            exec(compile(source,'cpu-analysis-runtime-cell','exec'),scope)
        self.assertEqual(scope['EXECUTION_MODE'],'cpu_analysis')
        self.assertEqual(scope['DEVICE'],'cpu')
        self.assertFalse(scope['CUDA_TRACKING_CHECK_PASSED'])
        self.assertEqual(subprocess.run.call_count,1)
        cli.assert_called_once_with('doctor')

    def test_missing_reference_does_not_generate_a_fake_accuracy_report(self):
        source=next(s for s in self.code if s.startswith('TRACKING_REFERENCE_PATH ='))
        cli=Mock();exec(compile(source,'tracking-score-cell','exec'),{'cli':cli})
        cli.assert_not_called()

    def test_checkpoint_discovery_skips_datasets_hidden_folders_and_keeps_last_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for folder,name in [('model','candidate.pt'),('model','last.pt'),('datasets','candidate.pt'),('.private','candidate.pt')]:
                p=root/folder/name;p.parent.mkdir(exist_ok=True);p.write_bytes(b'fixture')
            candidates,last=self.input_scope(root)['find_project_checkpoints'](root)
            self.assertEqual(candidates,[root/'model'/'candidate.pt'])
            self.assertEqual(last,[root/'model'/'last.pt'])

    def test_ambiguous_candidates_do_not_silently_select_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('first','second'):
                p=root/name/'candidate.pt';p.parent.mkdir();p.write_bytes(b'fixture')
            with self.assertRaises(AssertionError):self.input_scope(root,execute_selection=True)

    def test_uploaded_checkpoint_hash_must_match_adjacent_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'candidate.pt').write_bytes(b'fixture-not-weights')
            (root/'result.json').write_text(json.dumps({'checkpoint_sha256':'a'*64}),encoding='utf-8')
            with self.assertRaises(AssertionError):self.input_scope(root,execute_selection=True)

    def test_selected_fixture_retains_exact_file_hash_without_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);blob=b'fixture-not-weights';expected=hashlib.sha256(blob).hexdigest()
            (root/'candidate.pt').write_bytes(blob)
            (root/'result.json').write_text(json.dumps({'checkpoint_sha256':expected,'steps':100}),encoding='utf-8')
            scope=self.input_scope(root,execute_selection=True)
            self.assertEqual(scope['CHECKPOINT_SHA256'],expected)
            self.assertEqual(scope['CHECKPOINT_PATH'],root/'candidate.pt');scope['cli'].assert_not_called()
