"""Ephemeral, no-secret GPU Function factory for the frozen video-analysis worker."""


def build_app(modal,image_id,name):
    import re
    if not re.fullmatch('im-[a-zA-Z0-9_-]{1,150}',image_id):raise ValueError('Pinned image required')
    app=modal.App(name,include_source=False)
    @app.function(image=modal.Image.from_id(image_id),gpu='T4',serialized=True,include_source=False,
                  cpu=(2,2),memory=(8192,8192),timeout=540,startup_timeout=45,retries=0,
                  min_containers=0,max_containers=1,buffer_containers=0,scaledown_window=5,
                  single_use_containers=True,secrets=[],volumes={},env={},block_network=True,
                  restrict_modal_access=True)
    def analyze(files,hashes):
        import hashlib
        import json
        from pathlib import Path
        import subprocess
        import sys
        import threading
        import time
        started=time.monotonic()
        def event(stage):
            print('TOUCHLINE_EVENT '+json.dumps({'stage':stage,'elapsed_s':round(time.monotonic()-started,3)}),flush=True)
        allowed={'code.zip','candidate.pt','video24.mp4','worker.py','check_cuda_tracking.py'}
        if (not isinstance(files,dict) or set(files)!=allowed or set(hashes)!=allowed or
            any(type(v) is not bytes for v in files.values()) or sum(map(len,files.values()))>256*1024*1024):
            return {'status':'failed','failure_code':'INPUT_INVALID'}
        work=Path('/work');work.mkdir(exist_ok=True)
        for name,data in files.items():
            if hashlib.sha256(data).hexdigest()!=hashes[name]:return {'status':'failed','failure_code':'INPUT_INVALID'}
            with (work/name).open('xb') as stream:stream.write(data)
        (work/'job.json').write_text(json.dumps({'hashes':hashes}),encoding='utf-8')
        event('inputs_verified')
        process=subprocess.Popen([sys.executable,'-I',str(work/'worker.py')],cwd=work,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        def read_events():
            # Fixed stage vocabulary only; warnings/tracebacks are discarded here.
            for line in iter(lambda:process.stderr.readline(2048),b''):
                if not line.startswith(b'TOUCHLINE_EVENT '):continue
                try:item=json.loads(line[len(b'TOUCHLINE_EVENT '):])
                except Exception:continue
                if isinstance(item,dict) and item.get('stage') in {'cuda','parity','analysis','export'}:event(item['stage'])
        monitor=threading.Thread(target=read_events,daemon=True);monitor.start()
        stopped=threading.Event()
        def heartbeat():
            while not stopped.wait(15):event('worker_alive')
        pulse=threading.Thread(target=heartbeat,daemon=True);pulse.start()
        event('worker_started')
        timer=threading.Timer(480,process.kill);timer.start()
        output=bytearray()
        try:
            while True:
                chunk=process.stdout.read(1024)
                if not chunk:break
                if len(chunk)>8192-len(output):
                    process.kill();process.wait()
                    return {'status':'failed','failure_code':'OUTPUT_LIMIT'}
                output.extend(chunk)
            process.wait(timeout=10)
        finally:
            timer.cancel()
            if process.poll() is None:process.kill();process.wait()
            stopped.set();monitor.join(timeout=2);pulse.join(timeout=2)
            process.stdout.close()
            process.stderr.close()
        event('worker_finished')
        try:receipt=json.loads(output)
        except Exception:return {'status':'failed','failure_code':'WORKER_PROTOCOL'}
        if not isinstance(receipt,dict):return {'status':'failed','failure_code':'WORKER_PROTOCOL'}
        if process.returncode!=0 or receipt.get('status')!='complete':
            return {'status':'failed','failure_code':receipt.get('failure_code','INTERNAL_ERROR'),
                    'stage':receipt.get('stage','unknown')}
        result=work/'result.zip'
        if not result.is_file() or result.stat().st_size>64*1024*1024:
            return {'status':'failed','failure_code':'OUTPUT_LIMIT'}
        return {'status':'complete','receipt':receipt,'result_zip':result.read_bytes()}
    return app,analyze
