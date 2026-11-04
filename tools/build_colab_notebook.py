"""Regenerate the thin Colab workflow. No training or downloads run by this script."""
import json
from pathlib import Path
import textwrap

cells=[]
def md(text):
    cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(text).strip()+'\n'})
def code(text):
    cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':textwrap.dedent(text).strip()+'\n'})

md('''
# Touchline — Colab dataset-to-model workflow

**Priority 1: train a person detector for football footage.** This notebook inventories real data, converts reviewed labels, freezes grouped splits, compares a pretrained detector with a bounded fine-tune, reloads and exports a candidate. It does not implement football movement analysis.

Run cells in order. Configure paths, source class mappings and review flags. GPU execution is not pre-verified; no output cells have been fabricated. Read `docs/COLAB_RUNBOOK.md` for gates and troubleshooting. Start with your existing labels or a bounded SoccerNet Tracking subset. SAM 3-assisted repair is optional; fine-tuning SAM 3 is separate.
''')
code('''
from pathlib import Path
import sys, subprocess, json, os, hashlib, shutil, datetime, platform

REPO_DIR = Path('/content/touchline-starter')
REPO_URL = ''  # Your repository URL; leave empty if the starter is already extracted here.
REPO_REF = ''  # Prefer a reviewed commit/tag. Empty uses the repository default branch.
if not (REPO_DIR / 'pyproject.toml').exists():
    if not REPO_URL:
        from google.colab import files
        import zipfile, stat
        uploaded = files.upload()  # Select the supplied Touchline repository starter ZIP.
        assert len(uploaded) == 1, 'Select the single Touchline repository starter ZIP.'
        zip_path = next(iter(uploaded))
        with zipfile.ZipFile(zip_path) as z:
            assert sum(i.file_size for i in z.infolist()) < 50_000_000, 'Expected a small code ZIP, not a dataset.'
            roots=[Path(i.filename).parent for i in z.infolist() if Path(i.filename).name=='pyproject.toml']
            assert len(roots)==1, 'Expected one project root in the code ZIP.'
            prefix=roots[0]
            targets=[]
            for item in z.infolist():
                member=Path(item.filename)
                assert member.is_relative_to(prefix), 'Unexpected ZIP root.'
                target=(REPO_DIR/member.relative_to(prefix)).resolve()
                assert target.is_relative_to(REPO_DIR.resolve()), 'Unexpected code ZIP path.'
                assert not stat.S_ISLNK(item.external_attr >> 16), 'Unexpected symlink.'
                assert not target.exists(), 'Do not overwrite an existing code file.'
                targets.append((item,target))
            for item,target in targets:
                if item.is_dir():target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(z.read(item))
    else:
        subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)
        if REPO_REF:
            subprocess.run(['git', 'checkout', REPO_REF], cwd=REPO_DIR, check=True)
assert sys.version_info >= (3, 12), 'This starter targets Python 3.12+; select a compatible runtime.'
os.chdir(REPO_DIR)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[data]', 'pycocotools'], check=True)
def activate_project_imports(project_dir):
    # A subprocess installation does not refresh the running kernel's import hook.
    import importlib
    import sys
    from pathlib import Path
    source_dir = Path(project_dir).resolve() / 'src'
    if not (source_dir / 'touchline' / '__init__.py').is_file():
        raise FileNotFoundError(f'Touchline source missing under {source_dir}; finish the code ZIP setup first.')
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    importlib.invalidate_caches()
    module = importlib.import_module('touchline')
    if not Path(module.__file__).resolve().is_relative_to(source_dir):
        raise RuntimeError('A different Touchline checkout is already imported; use the matching kernel/code checkout.')
    return module

touchline_package = activate_project_imports(REPO_DIR)
print('Notebook kernel imports Touchline from:', touchline_package.__file__)

def cli(*args):
    return subprocess.run([sys.executable, '-m', 'touchline', *map(str, args)], check=True, cwd=REPO_DIR)
print('Code ready:', REPO_DIR)
''')
md('''
## Runtime gate
Select a GPU runtime first. Keep a working Colab Torch/Torchvision pair. If import/NMS fails, install a compatible pair using the official PyTorch selector, restart, then re-run. Do not upgrade Torchvision independently. No CPU fallback is silently used for a long training run.
''')
code('''
import torch, torchvision
from torchvision.ops import nms
nms(torch.tensor([[0., 0., 10., 10.]]), torch.tensor([0.9]), 0.5)
assert torch.cuda.is_available(), 'GPU not available. Change Colab runtime type or defer training.'
nms(torch.tensor([[0., 0., 10., 10.]],device='cuda'), torch.tensor([0.9],device='cuda'), 0.5)
DEVICE = 'cuda'
runtime = {'python':sys.version, 'torch':str(torch.__version__), 'torchvision':str(torchvision.__version__),
           'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0),
           'vram_bytes':torch.cuda.get_device_properties(0).total_memory, 'platform':platform.platform()}
print(json.dumps(runtime, indent=2))
cli('preflight', '--vision')
''')
md('''
## Storage and run identity
Drive holds durable input/export artifacts; local `/content` holds the working release for faster training. Use fresh names. Large data is not bundled into the code export. Colab can reset: the current runner has a final checkpoint but no periodic interruption/resume support.
''')
code('''
USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
DATA_ROOT = Path('/content/drive/MyDrive/Touchline/datasets')
EXPORT_ROOT = Path('/content/drive/MyDrive/Touchline/model-runs')
RUN_ID = datetime.datetime.now(datetime.timezone.utc).strftime('person-%Y%m%dT%H%M%S%fZ')
WORK = Path('/content/touchline-runs') / RUN_ID
WORK.mkdir(parents=True, exist_ok=False)
(WORK/'runtime.json').write_text(json.dumps(runtime, indent=2))
freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True, capture_output=True, text=True).stdout
(WORK/'environment.txt').write_text(freeze)
source_hashes = {str(p.relative_to(REPO_DIR)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((REPO_DIR/'src').rglob('*.py'))}
(WORK/'source-hashes.json').write_text(json.dumps(source_hashes, indent=2))
git = subprocess.run(['git','rev-parse','HEAD'],cwd=REPO_DIR,capture_output=True,text=True)
(WORK/'code-revision.txt').write_text(git.stdout.strip() if git.returncode == 0 else 'unversioned starter; see source-hashes.json')
print('Working directory:', WORK, '\\nFree GB:', round(shutil.disk_usage('/content').free/1e9, 1))
''')
md('''
## Optional SoccerNet acquisition
Skip when local labelled archives are already available. The inspected tracking repository has train/test/challenge ZIPs; start with **train.zip only**, after checking intended-use terms. Public availability is not a blanket commercial licence. Protected raw broadcast video is a different resource and may be gated. No full-organization download occurs.
''')
code('''
DOWNLOAD_SOCCERNET = False
SOCCERNET_TERMS_CONFIRMED = False
if DOWNLOAD_SOCCERNET:
    assert SOCCERNET_TERMS_CONFIRMED, 'Review the exact dataset terms/access for this use first.'
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub'], check=True)
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.dataset_info('SoccerNet/SN-Tracking-2023', files_metadata=True)
    entry = next(s for s in info.siblings if s.rfilename == 'train.zip')
    assert entry.size is not None, 'Inspect the archive size before downloading.'
    assert shutil.disk_usage('/content').free > entry.size * 3, 'Insufficient headroom; use another disk or a smaller approved local subset.'
    archive = hf_hub_download(repo_id='SoccerNet/SN-Tracking-2023', repo_type='dataset',
                              filename='train.zip', revision=info.sha, local_dir='/content/soccernet-download')
    from touchline.data.manifest import sha256_file
    (WORK/'soccernet-download.json').write_text(json.dumps({'repo':'SoccerNet/SN-Tracking-2023',
        'revision':info.sha, 'filename':'train.zip', 'sha256':sha256_file(archive)}, indent=2))
    print('Downloaded selected archive:', archive)
else:
    print('Using existing local/Drive data; no SoccerNet download.')
''')
code('''
# Optional checked ZIP extraction; set paths only for the archive you intend to import.
ARCHIVE_PATH = ''
EXTRACT_DIR = Path('/content/soccernet-selected')
MAX_EXPANDED_GB = 60
if ARCHIVE_PATH:
    import zipfile, stat
    assert not EXTRACT_DIR.exists(), 'Use a fresh extraction directory.'
    with zipfile.ZipFile(ARCHIVE_PATH) as z:
        total = sum(i.file_size for i in z.infolist())
        assert total <= MAX_EXPANDED_GB * 10**9 and total < shutil.disk_usage('/content').free * .9
        for item in z.infolist():
            target = (EXTRACT_DIR/item.filename).resolve()
            assert target.is_relative_to(EXTRACT_DIR.resolve()), 'Unsafe ZIP path.'
            assert not stat.S_ISLNK(item.external_attr >> 16), 'Symlink in ZIP; inspect manually.'
        EXTRACT_DIR.mkdir(parents=True)
        z.extractall(EXTRACT_DIR)
    print('Extracted bytes:', total)
''')
md('''
## Inventory and source inspection
Set `DATA_ROOT` to the extracted data. Inventory is metadata only. Review class names/order, normalized versus pixel boxes, missing labels, true match/session grouping and source eligibility. For SoccerNet, inspect a few `gameinfo.ini`, `seqinfo.ini` and `gt/gt.txt` files, then fill the explicit track-class manifest. Do not treat the MOT placeholder columns as classes.
''')
code('''
assert DATA_ROOT.is_dir(), 'Set DATA_ROOT to an existing extracted dataset directory.'
cli('inventory', DATA_ROOT, '--output', WORK/'inventory.json')
print((WORK/'inventory.json').read_text()[:10000])
for filename in ('gameinfo.ini', 'seqinfo.ini', 'gt.txt'):
    import itertools
    for path in itertools.islice(DATA_ROOT.rglob(filename), 2):
        print('SOURCE:', path, '\\n', path.read_text(errors='replace')[:2500])
''')
md('''
## Reviewed ETL configuration
Use `examples/README.md` and copy a template to your data folder. The templates deliberately have review flags false. Fill the real source classes and independent groups; do not create a fake group per frame. SoccerNet needs explicit sequence splits and track classes from inspected metadata. Keep official test/challenge out of training and model selection. For an initial custom experiment using only official training matches, clearly mark the custom holdouts; do not claim an official benchmark.

For the inspected SoccerNet `[Sequence]` INI layout, `tools/prepare_soccernet_config.py` can draft the sequence/class map and gameID-grouped internal holdouts. Run it against the extracted training folder, inspect its report and raw overlays, and complete the reviewer/eligibility/completeness fields before staging. It never marks the dataset reviewed automatically.
''')
code('''
# Select ONE route after reviewing the real data.
CONFIG_PATH = Path('/content/drive/MyDrive/Touchline/datasets/etl-config.json')
SOCCERNET_CONFIG = ''  # e.g. '/content/drive/MyDrive/Touchline/datasets/soccernet-config.json'
if SOCCERNET_CONFIG:
    cli('stage-soccernet', SOCCERNET_CONFIG, '--output', WORK/'soccernet-staging')
    CONFIG_PATH = WORK/'soccernet-staging'/'etl-config.json'
assert CONFIG_PATH.is_file(), 'Create the reviewed configuration first; see examples/.'
RELEASE_DIR = WORK/'release'
cli('etl', CONFIG_PATH, '--output', RELEASE_DIR)
MANIFEST = RELEASE_DIR/'release.json'
cli('validate-dataset', MANIFEST, '--training-gate')
release = json.loads(MANIFEST.read_text())
print('Split images:', {s:sum(i['split']==s for i in release['samples']) for s in ('train','val','test')})
''')
code('''
# Visual target QA: run this, then inspect more hard cases before permitting training.
from PIL import Image, ImageDraw
from IPython.display import display
for sample in release['samples'][:12]:
    with Image.open(RELEASE_DIR/sample['image_path']) as im:
        image = im.convert('RGB')
    draw = ImageDraw.Draw(image)
    for obj in sample['objects']:
        draw.rectangle(obj['bbox_xyxy'], outline='red', width=3)
    image.thumbnail((960,540))
    print(sample['source_image_ref'], sample['split'], sample['capture_group_id'])
    display(image)
''')
md('''
## Baseline and two-step wiring check
Set flags only after overlay inspection. Download consent covers official Torchvision pretrained weights. Tiny training checks finite loss, gradients and bundle reload; it is not an accuracy result. Runtime/step caps are cooperative, not billing guarantees.
''')
code('''
LABEL_OVERLAYS_ACCEPTED = False
ALLOW_WEIGHT_DOWNLOAD = False
RUN_TRAINING = False
EVAL_MAX_IMAGES = 30  # None means ALL validation images for the actual selection gate.
if RUN_TRAINING:
    assert LABEL_OVERLAYS_ACCEPTED and ALLOW_WEIGHT_DOWNLOAD
    cli('evaluate-detector', MANIFEST, '--output', WORK/'baseline-val', '--device', DEVICE,
        '--allow-weight-download', *(['--max-images', EVAL_MAX_IMAGES] if EVAL_MAX_IMAGES else []))
    cli('train-detector', MANIFEST, '--output', WORK/'tiny-train', '--device', DEVICE,
        '--execution-backend', 'colab', '--parent-run-id', RUN_ID,
        '--epochs', 2, '--max-steps', 2, '--max-seconds', 120, '--allow-weight-download')
    tiny_result = json.loads((WORK/'tiny-train'/'result.json').read_text())
    assert tiny_result['steps'] == 2, 'The tiny gate requires two completed optimizer steps; inspect the time cap/log.'
    from touchline.adapters.torchvision import TorchvisionDetector
    tiny = TorchvisionDetector(WORK/'tiny-train'/'candidate.pt', DEVICE)
    sample = next(s for s in release['samples'] if s['split']=='val')
    from torchvision.transforms.functional import to_tensor
    with Image.open(RELEASE_DIR/sample['image_path']) as im:
        tensor = to_tensor(im.convert('RGB')).to(DEVICE)
    with torch.inference_mode():
        reloaded = tiny.model([tensor])[0]
    assert all(torch.isfinite(v).all() for v in reloaded.values())
    print('Tiny model reload/forward passed:', {k:tuple(v.shape) for k,v in reloaded.items()})
    del tiny, reloaded, tensor
    torch.cuda.empty_cache()
else:
    print('Training is off. Inspect overlays, then set the three explicit flags to run.')
''')
md('''
## Bounded candidate fine-tuning and held-out comparison
The first recipe trains RPN/ROI heads, keeps the pretrained backbone frozen and uses batch one FP32. The low-resolution baseline may miss small/distant players: compare AP/recall and field examples before selecting it. Run all validation images before making a release decision. Increase caps only after the tiny wiring check and a resource measurement.
''')
code('''
if RUN_TRAINING:
    cli('train-detector', MANIFEST, '--output', WORK/'candidate-train', '--device', DEVICE,
        '--execution-backend', 'colab', '--parent-run-id', RUN_ID,
        '--epochs', 2, '--max-steps', 100, '--max-seconds', 600, '--allow-weight-download')
    CANDIDATE = WORK/'candidate-train'/'candidate.pt'
    cli('evaluate-detector', MANIFEST, '--checkpoint', CANDIDATE, '--output', WORK/'candidate-val',
        '--device', DEVICE, *(['--max-images', EVAL_MAX_IMAGES] if EVAL_MAX_IMAGES else []))
    baseline = json.loads((WORK/'baseline-val'/'evaluation.json').read_text())
    candidate = json.loads((WORK/'candidate-val'/'evaluation.json').read_text())
    assert baseline['sample_ids'] == candidate['sample_ids'], 'Evaluation samples differ.'
    assert baseline['manifest_sha256'] == candidate['manifest_sha256']
    display({'baseline':baseline['metrics'], 'candidate':candidate['metrics'],
             'subset_only':candidate['subset'], 'automatic_promotion':False})
''')
code('''
# Candidate prediction overlay after a download-free reload.
if RUN_TRAINING:
    from touchline.adapters.torchvision import TorchvisionDetector
    loaded = TorchvisionDetector(CANDIDATE, DEVICE)
    for sample in [s for s in release['samples'] if s['split']=='val'][:3]:
        with Image.open(RELEASE_DIR/sample['image_path']) as im:
            image=im.convert('RGB')
        with torch.inference_mode():
            pred=loaded.model([to_tensor(image).to(DEVICE)])[0]
        draw=ImageDraw.Draw(image)
        for box,score in zip(pred['boxes'].cpu().tolist(),pred['scores'].cpu().tolist()):
            if score>=0.5:draw.rectangle(box,outline='lime',width=3)
        image.thumbnail((960,540));display(image)
    del loaded
    torch.cuda.empty_cache()
''')
md('''
## Optional final test — after model selection is frozen
Use validation for all model/threshold decisions. Test is a final report, not another tuning set. This cell evaluates the chosen candidate only; if the pretrained model is selected, run the same evaluator without `--checkpoint` and with explicit download consent instead. Do not call any of these detection metrics validated metres, speeds or roster identity.
''')
code('''
RUN_FINAL_TEST = False
SELECTED_CANDIDATE_FROZEN = False
if RUN_FINAL_TEST:
    assert RUN_TRAINING and SELECTED_CANDIDATE_FROZEN
    cli('evaluate-detector', MANIFEST, '--checkpoint', CANDIDATE, '--split', 'test',
        '--output', WORK/'selected-test', '--device', DEVICE)
''')
md('''
## Export reproducible candidate artifacts
Copy artifacts and the corresponding image release separately to durable storage and verify hashes. The small model-artifact ZIP excludes images. Keep the model labelled **candidate** until its actual evaluation and field-domain checks pass. Keep the pretrained rollback.
''')
code('''
if RUN_TRAINING:
    from touchline.data.manifest import sha256_file
    artifact = WORK/'export'
    artifact.mkdir(exist_ok=False)
    for name in ('runtime.json','environment.txt','source-hashes.json','code-revision.txt','inventory.json'):
        shutil.copy2(WORK/name, artifact/name)
    for name in ('tiny-train','candidate-train','baseline-val','candidate-val','selected-test','soccernet-staging'):
        if (WORK/name).exists():shutil.copytree(WORK/name,artifact/name)
    shutil.copy2(MANIFEST, artifact/'release.json')
    shutil.copy2(RELEASE_DIR/'etl-report.json', artifact/'etl-report.json')
    hashes={str(p.relative_to(artifact)):sha256_file(p) for p in artifact.rglob('*') if p.is_file()}
    (artifact/'SHA256.json').write_text(json.dumps(hashes,indent=2))
    EXPORT_ROOT.mkdir(parents=True,exist_ok=True)
    destination=EXPORT_ROOT/RUN_ID
    assert not destination.exists(), 'Do not overwrite a previous model release.'
    shutil.copytree(artifact,destination)
    assert all(sha256_file(destination/name)==digest for name,digest in hashes.items())
    durable_release=EXPORT_ROOT.parent/'dataset-releases'/RUN_ID
    assert not durable_release.exists(), 'Use a fresh durable dataset path.'
    shutil.copytree(RELEASE_DIR,durable_release)
    from touchline.data.manifest import read_manifest
    read_manifest(durable_release/'release.json',training=True)
    archive=shutil.make_archive(str(WORK/'touchline-model-candidate'),'zip',artifact)
    print('Verified durable artifacts:', destination, '\\nLocal ZIP:', archive)
    print('Verified durable dataset release:', durable_release)
''')
md('''
## Optional SAM 3 and next gate
Do not block this detector path on SAM 3 installation. In a separately tested environment, use the official Meta repository and a local permitted checkpoint, then `touchline sam3-propose IMAGE --checkpoint LOCAL_WEIGHT --output proposals.json`. Correct proposals visually and export reviewed labels before ETL. Full SAM 3 fine-tuning remains a separate capped benchmark.

After the first real trained/evaluated/reloaded model bundle exists, proceed to the video/analysis tickets. Record actual versions, data/model hashes, failures and review decisions in the linked ticket completion record. This notebook does not claim a completed field trial or Level 1/2 validation.
''')

root=Path(__file__).resolve().parents[1]
target=root/'notebooks'/'01_dataset_to_model.ipynb'
target.parent.mkdir(parents=True,exist_ok=True)
for index,cell in enumerate(cells):
    cell['id']=f'touchline-{index:03d}'
target.write_text(json.dumps({'cells':cells,'metadata':{'colab':{'name':target.name},
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5},indent=2)+'\n',encoding='utf-8')
print(target)
