import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from touchline.settings import parse_operator_text,settings_from_mapping,load_operator_settings
from touchline.artifacts import inspect_colab_export
from touchline.data.manifest import sha256_file


class SettingsTests(unittest.TestCase):
    def test_custom_endpoint_url_is_restricted_and_proxy_keys_remain_private(self):
        settings=settings_from_mapping({'TOUCHLINE_DETECTOR_ENDPOINT_URL':'https://fixture.modal.run',
            'MODAL_PROXY_TOKEN_ID':'fixture-id','MODAL_PROXY_TOKEN_SECRET':'fixture-secret'},root=Path.cwd())
        self.assertEqual(settings.detector_endpoint_url,'https://fixture.modal.run')
        self.assertNotIn('fixture-secret',repr(settings))
        for url in ('http://fixture.modal.run','https://evil.invalid','https://user:pass@fixture.modal.run','https://fixture.modal.run?token=secret'):
            with self.assertRaises(ValueError):settings_from_mapping({'TOUCHLINE_DETECTOR_ENDPOINT_URL':url},root=Path.cwd())
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_artifact_paths_are_separate_from_provider_keys(self):
        settings=settings_from_mapping({'TOUCHLINE_CHECKPOINT_PATH':'export/candidate-train/candidate.pt',
            'TOUCHLINE_DATASET_MANIFEST':'dataset/release.json','META_API_KEY':'fixture-private-token'},root=self.root)
        self.assertEqual(settings.checkpoint_path,self.root/'export/candidate-train/candidate.pt')
        self.assertEqual(settings.dataset_manifest,self.root/'dataset/release.json')
        self.assertNotIn('fixture-private-token',repr(settings)+repr(settings.credentials)+json.dumps(settings.public_readiness()))
        self.assertFalse(settings.public_readiness()['paid_execution_enabled'])

    def test_loader_reads_only_injected_text_and_known_process_names(self):
        selected=self.root/'not-an-actual-secret-file'
        text='META_API_KEY=fixture-file\nTOUCHLINE_CHECKPOINT_PATH=export/candidate.pt'
        with patch('pathlib.Path.read_text',side_effect=AssertionError('No secret file may be read')):
            settings=load_operator_settings(selected,environ={'META_API_KEY':'fixture-process','UNRELATED_PRIVATE':'not-consumed'},reader=lambda p:text)
        self.assertEqual(settings.credentials.get('META_API_KEY'),'fixture-process')
        self.assertEqual(settings.checkpoint_path,self.root/'export/candidate.pt')

    def test_wrong_path_fails_before_reader(self):
        with self.assertRaises(ValueError):load_operator_settings(Path('relative-file'),environ={},reader=lambda p:self.fail('Reader called'))

    def test_no_shell_expansion_duplicate_or_unknown_fields(self):
        for value in ('META_API_KEY=a\nMETA_API_KEY=b','UNKNOWN_KEY=fixture','META_API_KEY=$(command)','META_API_KEY=${VALUE}',"META_API_KEY='unfinished"):
            with self.assertRaises(ValueError):parse_operator_text(value)
        self.assertEqual(parse_operator_text('# comment\nMETA_API_KEY="fixture#literal"')['META_API_KEY'],'fixture#literal')

    def test_manifest_is_not_checkpoint_and_drive_link_is_not_a_path(self):
        for values in ({'TOUCHLINE_CHECKPOINT_PATH':'release.json'},
                       {'TOUCHLINE_CHECKPOINT_PATH':'https://drive.google.com/file'},
                       {'TOUCHLINE_DATASET_MANIFEST':'candidate.pt'},
                       {'TOUCHLINE_CHECKPOINT_PATH':'~/candidate.pt'}):
            with self.assertRaises(ValueError):settings_from_mapping(values,root=self.root)

    def test_no_endpoint_exfiltration_alias_or_activation(self):
        for values in ({'META_BASE_URL':'https://example.invalid/v1'},{'BASETEN_MODEL':'other-model'},
                       {'TOUCHLINE_TRAINING_ENABLED':'true'},{'TOUCHLINE_AUTOFIX_ENABLED':'true'}):
            with self.assertRaises(ValueError):settings_from_mapping(values,root=self.root)

    def test_cli_uses_checkpoint_and_device_from_mocked_settings(self):
        from touchline.cli import main
        checkpoint=self.root/'candidate.pt';checkpoint.write_bytes(b'fixture')
        settings=settings_from_mapping({'TOUCHLINE_CHECKPOINT_PATH':str(checkpoint),'TOUCHLINE_DEVICE':'cuda'},root=self.root)
        source=self.root/'source.mp4';source.write_bytes(b'fixture')
        with patch('touchline.settings.load_operator_settings',return_value=settings),patch('touchline.analysis_policy.preflight_analysis_video',return_value={'fixture':True}),patch('touchline.adapters.torchvision.TorchvisionDetector') as detector,patch('touchline.pipeline.run_pipeline',return_value={'fixture':True}),redirect_stdout(io.StringIO()):
            self.assertEqual(main(['analyze',str(source),'--env-file',str(self.root/'not-an-actual-secret-file'),'--output',str(self.root/'run')]),0)
        self.assertEqual(detector.call_args.args[:2],(checkpoint,'cuda'))
        self.assertFalse(detector.call_args.kwargs['allow_download'])

    def test_studio_uses_selected_artifact_settings(self):
        from touchline.cli import main
        settings=settings_from_mapping({},root=self.root)
        with patch('touchline.settings.load_operator_settings',return_value=settings),patch('touchline.studio.run_studio_server') as server:
            main(['studio','--env-file',str(self.root/'not-an-actual-secret-file')])
        self.assertEqual(server.call_args.args[0],self.root/'runs')
        self.assertIs(server.call_args.kwargs['settings'],settings)


class ExportTests(unittest.TestCase):
    def setUp(self):
        from test_dataset import fixture
        from test_model_handoff import ModelHandoffTests
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.export=self.root/'export';train=self.export/'candidate-train';train.mkdir(parents=True)
        self.release=fixture()
        self.export.joinpath('release.json').write_text(json.dumps(self.release),encoding='utf-8')
        self.weights=train/'candidate.pt';self.weights.write_bytes(b'synthetic checkpoint; never unpickled')
        sample=ModelHandoffTests();sample.setUp();self.recipe=copy.deepcopy(sample.recipe)
        self.recipe.update(release_id=self.release['release_id'],dataset_version=self.release['dataset_version'],dataset_manifest_sha256=sha256_file(self.export/'release.json'))
        self.result={**sample.result,'checkpoint_sha256':sha256_file(self.weights)}
        (train/'recipe.json').write_text(json.dumps(self.recipe),encoding='utf-8')
        (train/'result.json').write_text(json.dumps(self.result),encoding='utf-8')
        self.rehash()
    def tearDown(self):self.tmp.cleanup()
    def rehash(self):
        names=['release.json','candidate-train/candidate.pt','candidate-train/recipe.json','candidate-train/result.json']
        (self.export/'SHA256.json').write_text(json.dumps({n:sha256_file(self.export/n) for n in names}),encoding='utf-8')

    def test_known_export_format_passes_without_loading_weights_or_images(self):
        report=inspect_colab_export(self.export)
        self.assertFalse(report['weights_loaded']);self.assertFalse(report['images_verified'])
        self.assertEqual(report['sample_count'],3)
        self.assertEqual(report['runtime_compatibility'],'not checked')

    def test_copied_manifest_is_not_claimed_as_complete_image_dataset(self):
        with self.assertRaises(ValueError):inspect_colab_export(self.export,verify_images=True)
        with self.assertRaises(ValueError):inspect_colab_export(self.export,dataset_manifest=self.export/'release.json',verify_images=True)

    def test_changed_checkpoint_hash_is_rejected(self):
        self.weights.write_bytes(b'changed')
        with self.assertRaises(ValueError):inspect_colab_export(self.export)

    def test_wrong_dataset_release_is_rejected(self):
        selected=self.root/'dataset'/'release.json';selected.parent.mkdir();selected.write_text('{}',encoding='utf-8')
        with self.assertRaises(ValueError):inspect_colab_export(self.export,dataset_manifest=selected)

    def test_recipe_dataset_mismatch_is_rejected_even_if_hash_index_updated(self):
        self.recipe['dataset_manifest_sha256']='a'*64
        (self.export/'candidate-train/recipe.json').write_text(json.dumps(self.recipe),encoding='utf-8');self.rehash()
        with self.assertRaises(ValueError):inspect_colab_export(self.export)

    def test_cli_export_check_does_not_invoke_env_loader(self):
        from touchline.cli import main
        with patch('touchline.settings.load_operator_settings',side_effect=AssertionError('No credential read')),redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['export-check',str(self.export)]),0)
        self.assertEqual(json.loads(output.getvalue())['state'],'export_integrity_passed_candidate_unvalidated')
