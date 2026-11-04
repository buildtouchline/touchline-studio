"""Explicit operator configuration. Importing this module reads no environment/files.

The agent must never invoke the loader on an actual credential file/environment.
Automated tests use synthetic mappings and an injected in-memory file reader.
"""
from dataclasses import dataclass, field
from pathlib import Path
import os
import re
from urllib.parse import urlsplit
from .diagnostics import diagnosed

SECRET_NAMES=frozenset({'BASETEN_API_KEY','META_API_KEY','MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','FIREWORKS_API_KEY','HF_TOKEN','SARVAM_API_KEY','MODAL_PROXY_TOKEN_ID','MODAL_PROXY_TOKEN_SECRET'})
PUBLIC_NAMES=frozenset({'TOUCHLINE_CHECKPOINT_PATH','TOUCHLINE_DATASET_MANIFEST','TOUCHLINE_EXPORT_DIR',
    'TOUCHLINE_RUNS_DIR','TOUCHLINE_DEVICE','TOUCHLINE_MODAL_VOLUME_NAME',
    'BASETEN_BASE_URL','BASETEN_MODEL','META_BASE_URL','META_SAM_MODEL',
    'TOUCHLINE_MODAL_APP_NAME','TOUCHLINE_BASETEN_SECRET_NAME','TOUCHLINE_META_SAM_SECRET_NAME',
    'TOUCHLINE_TRAINING_ENABLED','TOUCHLINE_AUTOFIX_ENABLED','TOUCHLINE_DETECTOR_ENDPOINT_URL'})
ALLOWED_NAMES=SECRET_NAMES|PUBLIC_NAMES


def parse_operator_text(text):
    """A deliberately small dotenv subset: no expansion, multiline or shell syntax."""
    if not isinstance(text,str) or len(text)>65536:raise ValueError('Invalid settings text')
    values={}
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        if '=' not in line:raise ValueError('Settings require NAME=value lines')
        name,value=line.split('=',1);name=name.strip();value=value.strip()
        if name not in ALLOWED_NAMES or name in values:raise ValueError('Unknown or duplicate setting')
        if value[:1] in ('"',"'"):
            if len(value)<2 or value[-1]!=value[0]:raise ValueError('Unterminated quoted setting')
            value=value[1:-1]
        if any(c in value for c in ('\x00','\n','\r')) or '${' in value or '$(' in value:
            raise ValueError('Expansion/control characters are unsupported')
        values[name]=value
    return values


@dataclass(frozen=True,repr=False)
class Credentials:
    _values: dict[str,str]=field(default_factory=dict,repr=False)
    def __repr__(self):return 'Credentials([redacted])'
    def get(self,name):
        if name not in SECRET_NAMES:raise ValueError('Unknown credential name')
        return self._values.get(name,'')


@dataclass(frozen=True,repr=False)
class RuntimeSettings:
    root: Path
    checkpoint_path: Path|None
    dataset_manifest: Path|None
    export_dir: Path|None
    runs_dir: Path
    device: str
    modal_volume_name: str
    baseten_base_url: str
    baseten_model: str
    meta_base_url: str
    meta_sam_model: str
    credentials: Credentials=field(repr=False)
    detector_endpoint_url: str|None=None
    def __repr__(self):return 'RuntimeSettings([private settings])'
    def public_readiness(self):
        return {'schema':'touchline.settings.v1','checkpoint_configured':self.checkpoint_path is not None,
            'dataset_manifest_configured':self.dataset_manifest is not None,
            'export_configured':self.export_dir is not None,'device':self.device,
            'credentials':'not displayed','paid_execution_enabled':False,
            'training_enabled':False,'autofix_enabled':False}


@diagnosed('config')
def settings_from_mapping(values,*,root):
    root=Path(root)
    if not root.is_absolute():raise ValueError('Explicit absolute project root required')
    for name in values:
        if name not in ALLOWED_NAMES or not isinstance(values[name],str):raise ValueError('Unknown or non-string setting')
        if len(values[name])>16384 or any(c in values[name] for c in ('\x00','\n','\r')):
            raise ValueError('Invalid setting length/control characters')
    def path(name,default=None):
        value=values.get(name,default)
        if not value:return None
        if '://' in value:raise ValueError('Artifact settings require runtime filesystem paths, not URLs')
        if value.startswith('~'):raise ValueError('Use an explicit path instead of home expansion')
        p=Path(value)
        return (p if p.is_absolute() else root/p).resolve()
    checkpoint=path('TOUCHLINE_CHECKPOINT_PATH')
    manifest=path('TOUCHLINE_DATASET_MANIFEST')
    if checkpoint is not None and checkpoint.suffix.lower() not in ('.pt','.pth'):
        raise ValueError('The checkpoint must be trained .pt/.pth weights, not release.json')
    if manifest is not None and manifest.name!='release.json':raise ValueError('Dataset setting must select release.json')
    if any(values.get(k,'false').lower()!='false' for k in ('TOUCHLINE_TRAINING_ENABLED','TOUCHLINE_AUTOFIX_ENABLED')):
        raise ValueError('Training/autofix activation requires the bounded controller, not an env toggle')
    pinned={'BASETEN_BASE_URL':'https://inference.baseten.co/v1','BASETEN_MODEL':'meta-models/muse-spark-open-1026',
            'META_BASE_URL':'https://api.meta.ai/v1','META_SAM_MODEL':'sam-3.1'}
    for key,expected in pinned.items():
        if values.get(key,expected).rstrip('/')!=expected:raise ValueError('Provider endpoint/model must match the approved contract')
    device=values.get('TOUCHLINE_DEVICE','cpu')
    if device not in ('cpu','cuda'):raise ValueError('Device must be cpu or cuda')
    volume=values.get('TOUCHLINE_MODAL_VOLUME_NAME','touchline-artifacts')
    if not re.fullmatch('[a-z0-9][a-z0-9-]{1,62}',volume):raise ValueError('Invalid Volume name')
    endpoint=values.get('TOUCHLINE_DETECTOR_ENDPOINT_URL') or None
    if endpoint is not None:
        parsed=urlsplit(endpoint)
        if (parsed.scheme!='https' or not parsed.hostname or not parsed.hostname.endswith('.modal.run') or
            parsed.netloc!=parsed.hostname or parsed.query or parsed.fragment or '..' in parsed.path):
            raise ValueError('Use the approved HTTPS Modal web-app URL without credentials/query')
    return RuntimeSettings(root.resolve(),checkpoint,manifest,path('TOUCHLINE_EXPORT_DIR'),path('TOUCHLINE_RUNS_DIR','runs'),device,volume,
        pinned['BASETEN_BASE_URL'],pinned['BASETEN_MODEL'],pinned['META_BASE_URL'],pinned['META_SAM_MODEL'],
        Credentials({k:values[k] for k in SECRET_NAMES if values.get(k)}),detector_endpoint_url=endpoint)


@diagnosed('config')
def load_operator_settings(env_file,*,environ=None,reader=None):
    """OWNER-ONLY runtime entry point; exact absolute file, no discovery/search.

    Process variables win only for known names. Caller-selected file parent is
    the root for relative artifact paths, independent of notebook/terminal CWD.
    Tests must inject both reader and synthetic environ; never call on real keys.
    """
    selected=Path(env_file)
    if not selected.is_absolute():raise ValueError('Choose one explicit absolute settings file')
    for p in (selected,*selected.parents):
        if p.is_symlink():raise ValueError('Symlink settings paths are forbidden')
    text=reader(selected) if reader is not None else selected.read_text(encoding='utf-8')
    values=parse_operator_text(text)
    process=os.environ if environ is None else environ
    for name in ALLOWED_NAMES:
        if name in process:values[name]=process[name]
    return settings_from_mapping(values,root=selected.parent)
