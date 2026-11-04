# Colab input cell. Read only known model metadata under the selected project root.
from pathlib import Path
import os
import json
import hashlib
from google.colab import drive

drive.mount('/content/drive')
TOUCHLINE_ROOT = Path('/content/drive/MyDrive/Touchline')
assert TOUCHLINE_ROOT.is_dir(), 'Mount the MyDrive/Touchline project folder first.'

def find_project_checkpoints(root, max_entries=20000):
    root = Path(root).resolve()
    candidates = []
    last_files = []
    inspected = 0
    excluded = {'datasets', 'dataset-releases', 'images', 'img1', 'labels', 'videos', 'analysis-runs'}
    for folder, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if not d.startswith('.') and d not in excluded
                                and not (Path(folder)/d).is_symlink())
        inspected += len(directories) + len(files)
        if inspected > max_entries:
            raise ValueError('Model search limit reached. Set CHECKPOINT_OVERRIDE to the exact model file.')
        for name in files:
            if name not in ('candidate.pt', 'last.pt'):
                continue
            path = Path(folder)/name
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('Checkpoint must be inside the selected project folder without symlinks.')
            (candidates if name == 'candidate.pt' else last_files).append(path)
    return sorted(candidates), sorted(last_files)

def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

CHECKPOINT_OVERRIDE = None  # Set an exact Path here if multiple candidate.pt files are listed.
VIDEO_PATH = None  # Exact analyzed 24 FPS landscape/panoramic video for inference or visual review.
EXPORT_DIR = None  # Optional complete original export root, including SHA256.json and release.json.
DRIVE_OUTPUT_ROOT = TOUCHLINE_ROOT/'analysis-runs'
if globals().get('EXECUTION_MODE','cuda_analysis') == 'cpu_replay':
    CHECKPOINT_PATH = CHECKPOINT_SHA256 = None
    print('CPU replay needs no checkpoint; select the saved-run folder in step 3a.')
else:
    if CHECKPOINT_OVERRIDE is None:
        candidate_files, last_files = find_project_checkpoints(TOUCHLINE_ROOT)
        print('candidate.pt files:', [str(p.relative_to(TOUCHLINE_ROOT)) for p in candidate_files])
        print('last.pt files:', [str(p.relative_to(TOUCHLINE_ROOT)) for p in last_files])
        assert len(candidate_files) == 1, 'Finish upload or select one exact CHECKPOINT_OVERRIDE; never choose an arbitrary model.'
        CHECKPOINT_PATH = candidate_files[0]
    else:
        CHECKPOINT_PATH = Path(CHECKPOINT_OVERRIDE)
    assert CHECKPOINT_PATH.is_file() and CHECKPOINT_PATH.suffix.lower() in ('.pt', '.pth')
    assert not any(p.is_symlink() for p in (CHECKPOINT_PATH, *CHECKPOINT_PATH.absolute().parents))
    assert CHECKPOINT_PATH.resolve().is_relative_to(TOUCHLINE_ROOT.resolve())
    CHECKPOINT_SHA256 = checkpoint_digest(CHECKPOINT_PATH)
    result_path = CHECKPOINT_PATH.parent/'result.json'
    if result_path.is_file():
        result_metadata = json.loads(result_path.read_text(encoding='utf-8'))
        expected = result_metadata.get('checkpoint_sha256')
        assert isinstance(expected, str) and expected == CHECKPOINT_SHA256, 'Checkpoint differs from result.json; finish upload/check the chosen file.'
        print('Training result:', {k: result_metadata.get(k) for k in ('status','steps','run_id','promotion_state')})
    recipe_path = CHECKPOINT_PATH.parent/'recipe.json'
    if recipe_path.is_file():
        recipe_metadata = json.loads(recipe_path.read_text(encoding='utf-8'))
        print('Recipe:', {k: recipe_metadata.get(k) for k in ('architecture','num_classes','class_map','torch','torchvision','release_id')})
    if EXPORT_DIR is not None:
        cli('export-check', EXPORT_DIR)
    print('Selected checkpoint:', CHECKPOINT_PATH)
    print('Checkpoint SHA256:', CHECKPOINT_SHA256)
    print('No weights loaded, inference started or provider keys accessed. Run the runtime check before analysis.')
