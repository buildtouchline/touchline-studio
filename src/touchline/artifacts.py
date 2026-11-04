"""Inspect the actual Colab export layout without loading/unpickling weights."""
import json
from pathlib import Path
from .adapters.handoff import validate_candidate_metadata
from .data.manifest import read_manifest,sha256_file
from .diagnostics import diagnosed


def _file(root,relative):
    original=Path(root).absolute()
    if any(p.is_symlink() for p in (original,*original.parents)):raise ValueError('Unsafe artifact directory')
    root=original.resolve();path=root/relative
    if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Unsafe artifact path')
    if not path.is_file():raise ValueError('Required Colab export artifact is missing')
    return path


@diagnosed('artifacts')
def inspect_colab_export(export_dir,*,dataset_manifest=None,verify_images=False):
    """Check known files/hashes/lineage; never infer compatibility from .pt suffix.

    Embedded checkpoint metadata/runtime compatibility is still checked by the
    existing weights_only=True Torchvision loader when the operator runs inference.
    """
    root=Path(export_dir)
    checkpoint=_file(root,'candidate-train/candidate.pt')
    recipe_path=_file(root,'candidate-train/recipe.json')
    result_path=_file(root,'candidate-train/result.json')
    copied_manifest=_file(root,'release.json')
    hashes=json.loads(_file(root,'SHA256.json').read_text(encoding='utf-8'))
    if not isinstance(hashes,dict):raise ValueError('Export hash index must be an object')
    for relative in ('candidate-train/candidate.pt','candidate-train/recipe.json','candidate-train/result.json','release.json'):
        if hashes.get(relative)!=sha256_file(_file(root,relative)):raise ValueError('Export artifact hash mismatch')
    recipe=json.loads(recipe_path.read_text(encoding='utf-8'))
    result=json.loads(result_path.read_text(encoding='utf-8'))
    validate_candidate_metadata(recipe,runtime_torch=recipe.get('torch'),runtime_torchvision=recipe.get('torchvision'))
    manifest_hash=sha256_file(copied_manifest)
    if recipe['dataset_manifest_sha256']!=manifest_hash:raise ValueError('Checkpoint recipe and release lineage disagree')
    if result.get('checkpoint_sha256')!=hashes['candidate-train/candidate.pt']:
        raise ValueError('Training result and checkpoint hashes disagree')
    if result.get('status') not in ('capped','completed') or type(result.get('steps')) is not int or result['steps']<1:
        raise ValueError('Training result does not evidence an exported candidate')
    if result.get('promotion_state')!='unvalidated_candidate':raise ValueError('Unexpected candidate status')
    copied=read_manifest(copied_manifest,training=False)
    if any(recipe.get(k)!=copied.get(k) for k in ('release_id','dataset_version','class_map')):
        raise ValueError('Recipe and dataset identities disagree')
    images_verified=False
    if dataset_manifest is not None:
        selected=Path(dataset_manifest)
        if selected.name!='release.json' or any(p.is_symlink() for p in (selected,*selected.parents)):
            raise ValueError('Select the complete dataset release manifest')
        if sha256_file(selected)!=manifest_hash:raise ValueError('Selected dataset release does not match the trained artifact')
        read_manifest(selected,training=verify_images)
        images_verified=verify_images
    elif verify_images:
        raise ValueError('Image verification needs the separate full dataset manifest')
    return {'schema':'touchline.colab-artifacts.v1','checkpoint_sha256':hashes['candidate-train/candidate.pt'],
        'dataset_manifest_sha256':manifest_hash,'architecture':recipe['architecture'],
        'class_map':recipe['class_map'],'sample_count':len(copied['samples']),
        'dataset_selected':dataset_manifest is not None,'images_verified':images_verified,
        'weights_loaded':False,'runtime_compatibility':'not checked','accuracy_95_percent':'not validated',
        'state':'export_integrity_passed_candidate_unvalidated'}
