"""Explicit runtime-only key loading: metadata or one capped authorized Hello.

Never print credentials, headers, response bodies or exception text. This is not
provider billing/quality acceptance. No compute/training job is launched.
Offline tests inject fake clients/settings.
"""
import argparse
from contextlib import redirect_stdout,redirect_stderr,closing
import json
import logging
import os
import sqlite3
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
MAX_BODY=262144
CATALOGS={
    'meta':('https://api.meta.ai/v1/models','META_API_KEY',('sam-3.1','muse-voice-transcribe-1.0')),
    'baseten':('https://inference.baseten.co/v1/models','BASETEN_API_KEY',('meta-models/muse-spark-open-1026',)),
}


def probe_catalog(client,url,key,models):
    if not key:return {'state':'key_missing'}
    try:
        with client.stream('GET',url,headers={'Authorization':'Bearer '+key}) as response:
            status=response.status_code
            if status!=200:return {'state':'http_rejected','http_status':status}
            body=bytearray()
            for chunk in response.iter_bytes():
                if len(chunk)>MAX_BODY-len(body):return {'state':'response_too_large','http_status':status}
                body.extend(chunk)
            data=json.loads(body)
        rows=data.get('data') if isinstance(data,dict) else None
        if not isinstance(rows,list):return {'state':'unexpected_schema','http_status':status}
        ids={row.get('id') for row in rows if isinstance(row,dict) and isinstance(row.get('id'),str)}
        return {'state':'catalog_received','http_status':status,
                'requested_models':{model:model in ids for model in models}}
    except Exception:
        return {'state':'connection_or_response_failed'}


def check_access(settings,client):
    results={}
    for provider,(url,name,models) in CATALOGS.items():
        results[provider]=probe_catalog(client,url,settings.credentials.get(name),models)
    results['sarvam']={'state':'key_configured_not_authenticated' if settings.credentials.get('SARVAM_API_KEY') else 'key_missing'}
    results['modal']={'state':'tokens_configured_not_authenticated' if
        all(settings.credentials.get(k) for k in ('MODAL_TOKEN_ID','MODAL_TOKEN_SECRET')) else 'token_pair_missing'}
    return {'schema':'touchline.provider-access.v1','results':results,'paid_requests':0,
            'metadata_requests_maximum':2,'secret_values':'never emitted',
            'inference_and_training':'not executed','billing':'not inspected'}


class Discard:
    def write(self,value):return len(value)
    def flush(self):pass


def probe_modal(settings,factory):
    tokens=[settings.credentials.get(k) for k in ('MODAL_TOKEN_ID','MODAL_TOKEN_SECRET')]
    if not all(tokens):return {'state':'token_pair_missing','jobs_launched':0}
    try:
        # Suppress any SDK/server notices; emit only this fixed status schema.
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            client=factory(*tokens)
            client.hello()
        return {'state':'authenticated_hello','jobs_launched':0}
    except Exception:
        return {'state':'authentication_or_connection_failed','jobs_launched':0}


def probe_baseten_hello(settings,factory):
    key=settings.credentials.get('BASETEN_API_KEY')
    if not key:return {'state':'key_missing'}
    try:
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            with factory(api_key=key,base_url='https://inference.baseten.co/v1',max_retries=0,timeout=20.0) as client:
                result=client.chat.completions.create(model='meta-models/muse-spark-open-1026',
                    messages=[{'role':'user','content':'Hello!'}],max_tokens=32,stream=False)
        usage=getattr(result,'usage',None)
        counts={name:getattr(usage,name,None) for name in ('prompt_tokens','completion_tokens','total_tokens')}
        counts={k:v for k,v in counts.items() if type(v) is int and 0<=v<=10000000}
        return {'state':'completion_received','model':'meta-models/muse-spark-open-1026',
                'max_tokens':32,'usage':counts,'response_content':'not emitted'}
    except Exception as exc:
        status=getattr(exc,'status_code',None)
        if type(status) is int and 100<=status<=599:return {'state':'http_rejected','http_status':status}
        return {'state':'connection_or_response_failed'}


def reserve_single_smoke(path,*,retry_receipt=None):
    """Reserve the entire $1 authorization before one possible billed call.

    This receipt is never auto-released; billing reconciliation is independent.
    A duplicate launch is rejected even after a timeout/process interruption.
    """
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():raise ValueError('Unsafe smoke allowance path')
    with closing(sqlite3.connect(path,timeout=10)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('CREATE TABLE IF NOT EXISTS smoke (authorization TEXT PRIMARY KEY, reserved_usd TEXT NOT NULL)')
        if retry_receipt is None:
            db.execute('INSERT INTO smoke VALUES (?,?)',('user-20261101-one-dollar','1.00'))
        else:
            # An explicit owner retry may reuse the same retained reservation
            # only after the original request was definitively rejected at auth.
            previous=json.loads(retry_receipt.read_text(encoding='utf-8'))
            if previous.get('results',{}).get('baseten')!={'state':'http_rejected','http_status':401}:
                raise ValueError('Only the known prior 401 may permit the requested retry')
            if db.execute('SELECT reserved_usd FROM smoke WHERE authorization=?',('user-20261101-one-dollar',)).fetchone()!=('1.00',):
                raise ValueError('Original test reservation is missing')
            db.execute('CREATE TABLE IF NOT EXISTS smoke_retry (authorization TEXT PRIMARY KEY)')
            db.execute('INSERT INTO smoke_retry VALUES (?)',('user-20261101-one-dollar',))


def main(argv=None):
    logging.disable(logging.CRITICAL)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--modal-only',action='store_true')
    parser.add_argument('--baseten-hello',action='store_true')
    parser.add_argument('--baseten-only',action='store_true')
    parser.add_argument('--retry-after-401',action='store_true')
    args=parser.parse_args(argv)
    if sum((args.modal_only,args.baseten_hello,args.baseten_only))>1:raise ValueError('Select one probe mode')
    if args.retry_after_401 and not args.baseten_hello:raise ValueError('A retry requires Hello mode')
    if args.env_file!=ROOT/'.env' or args.env_file.is_symlink():
        raise ValueError('Only the explicitly authorized checkout settings file is allowed')
    if args.output.exists() or args.output.suffix.lower()!='.json':raise ValueError('Use a fresh JSON output')
    if not args.output.resolve().is_relative_to((ROOT/'runs').resolve()):raise ValueError('Keep receipts under runs')
    sys.path.insert(0,str(ROOT/'src'))
    from touchline.settings import load_operator_settings
    import httpx
    # No provider SDK/profile discovery. Ignore process variables: only the
    # operator-selected file supplies credentials, and values never leave runtime.
    settings=load_operator_settings(args.env_file,environ={})
    if args.baseten_only:
        url,name,models=CATALOGS['baseten']
        with httpx.Client(trust_env=False,follow_redirects=False,timeout=20.0) as client:
            result=probe_catalog(client,url,settings.credentials.get(name),models)
        report={'schema':'touchline.provider-access.v1','results':{'baseten':result},
                'paid_requests':0,'secret_values':'never emitted','billing':'not inspected'}
    elif args.baseten_hello:
        from openai import OpenAI
        if settings.credentials.get('BASETEN_API_KEY'):
            reserve_single_smoke(ROOT/'runs'/'provider-smoke-allowance-20261101.sqlite3',
                retry_receipt=ROOT/'runs'/'baseten-hello-20261101-01.json' if args.retry_after_401 else None)
        with httpx.Client(trust_env=False,follow_redirects=False,timeout=20.0) as transport:
            def factory(**kwargs):return OpenAI(**kwargs,http_client=transport)
            result=probe_baseten_hello(settings,factory)
        report={'schema':'touchline.provider-access.v1','results':{'baseten':result},
                'inference_requests_maximum':1,'max_output_tokens':32,
                'authorization_reserved_usd':'1.00','billing':'pending independent reconciliation',
                'secret_values':'never emitted'}
    elif args.modal_only:
        # Force a new absent profile path and the official API, so SDK imports
        # cannot discover/read the operator's real ~/.modal.toml or change host.
        profile=ROOT/'runs'/'metadata-only-no-profile.toml'
        if profile.exists() or profile.is_symlink():raise ValueError('Reserved no-profile path must be absent')
        os.environ['MODAL_CONFIG_PATH']=str(profile)
        os.environ['MODAL_SERVER_URL']='https://api.modal.com'
        os.environ['MODAL_LOGLEVEL']='CRITICAL'
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            import modal
        report={'schema':'touchline.provider-access.v1','results':{'modal':probe_modal(settings,modal.Client.from_credentials)},
                'paid_requests':0,'secret_values':'never emitted','billing':'not inspected'}
    else:
        with httpx.Client(trust_env=False,follow_redirects=False,timeout=20.0) as client:
            report=check_access(settings,client)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception:
        print('{"state":"settings_or_runtime_failed","secret_values":"never emitted"}')
        raise SystemExit(1) from None
