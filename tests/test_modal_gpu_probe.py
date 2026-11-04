import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS=Path(__file__).resolve().parents[1]/'tools'
with patch.object(sys,'path',[str(TOOLS),*sys.path]):
    spec=importlib.util.spec_from_file_location('touchline_gpu_probe_test',TOOLS/'probe_modal_gpu.py')
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class GpuProbeTests(unittest.TestCase):
    def test_error_classification_never_returns_private_provider_text(self):
        result=probe.safe_error(RuntimeError('fixture-secret: GPU sandbox billing rejected https://private.invalid'))
        self.assertIn('billing_mentioned',result['flags'])
        self.assertNotIn('fixture-secret',str(result))
        self.assertNotIn('private.invalid',str(result))
        self.assertFalse(result['raw_error_retained'])

    def test_internal_provider_failure_preserves_type_without_message(self):
        InternalFailure=type('InternalFailure',(Exception,),{})
        result=probe.safe_error(InternalFailure('fixture-private-value'))
        self.assertEqual(result['error_category'],'InternalFailure')
        self.assertNotIn('fixture-private-value',str(result))

    def test_gpu_output_is_single_device_and_fixed_metadata_only(self):
        self.assertEqual(probe.parse_gpu_output('Tesla T4, 15360 MiB, 570.124.06'),
            {'gpu_name':'T4','memory_mib':15360,'driver_version':'570.124.06'})
        for raw in ('H100, 80000 MiB, 570.1','T4, secret, 570.1',
                    'T4, 15360 MiB, 570.1\nT4, 15360 MiB, 570.1'):
            with self.assertRaises(ValueError):probe.parse_gpu_output(raw)

    def test_duplicate_probe_and_missing_approval_fail_before_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'runs/probe';work.mkdir(parents=True)
            (work/'gpu-probe-main-t4.json').write_text('{}',encoding='utf-8')
            with patch.object(probe,'ROOT',root),patch.object(probe,'modal_client') as client:
                with self.assertRaises(ValueError):probe.run(work,approved=False)
                with self.assertRaises(FileExistsError):probe.run(work,approved=True)
                client.assert_not_called()
