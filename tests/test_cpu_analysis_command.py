import importlib.util
from pathlib import Path
import unittest
import tempfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cpu_analysis_command_test',ROOT/'tools/run_cpu_video_analysis.py')
cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)


class CpuAnalysisCommandTests(unittest.TestCase):
    def test_full_run_uses_cpu_and_has_no_inference_cap_or_provider_settings(self):
        args=cpu.command(Path('video.mp4'),Path('candidate.pt'),Path('run'))
        self.assertEqual(args[args.index('--device')+1],'cpu')
        self.assertEqual(args[args.index('--tracker')+1],'bytetrack')
        self.assertEqual(args[args.index('--fps')+1],'6')
        self.assertNotIn('--max-selected',args)
        self.assertNotIn('--env-file',args)
        self.assertNotIn('--allow-weight-download',args)

    def test_local_child_caches_are_owned_and_provider_keys_are_not_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);env=cpu.cpu_environment(root)
            for name in ('MPLCONFIGDIR','TORCH_HOME','USERPROFILE','APPDATA','LOCALAPPDATA'):
                self.assertTrue(Path(env[name]).is_relative_to(root))
                self.assertTrue(Path(env[name]).is_dir())
            self.assertFalse(any('TOKEN' in k or 'SECRET' in k or 'API_KEY' in k for k in env))
