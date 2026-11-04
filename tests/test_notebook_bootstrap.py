"""Exercise the actual notebook import bootstrap in isolated Python processes."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest


class NotebookBootstrapTests(unittest.TestCase):
    def helper_source(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / 'notebooks/01_dataset_to_model.ipynb').read_text(encoding='utf-8'))
        cell = next(c for c in notebook['cells'] if c['cell_type'] == 'code')
        helper = next(n for n in ast.parse(cell['source']).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'activate_project_imports')
        return root, ast.unparse(helper)

    def test_running_kernel_can_import_source_without_editable_hook(self):
        root, helper = self.helper_source()
        script = helper + '\n' + '\n'.join([
            f'module = activate_project_imports({str(root)!r})',
            'from pathlib import Path',
            f'assert Path(module.__file__).resolve() == Path({str(root / "src/touchline/__init__.py")!r}).resolve()',
            'from touchline.data.manifest import sha256_file',
            'assert callable(sha256_file)',
        ])
        subprocess.run([sys.executable, '-I', '-c', script], check=True, capture_output=True, text=True)

    def test_missing_checkout_fails_before_importing_an_unrelated_package(self):
        root, helper = self.helper_source()
        script = helper + '\n' + '\n'.join([
            'try:',
            f'    activate_project_imports({str(root / "tests")!r})',
            'except FileNotFoundError:',
            '    pass',
            'else:',
            '    raise AssertionError("Missing checkout was accepted")',
        ])
        subprocess.run([sys.executable, '-I', '-c', script], check=True, capture_output=True, text=True)
