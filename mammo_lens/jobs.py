"""Single-worker, operator-authorized training jobs with persistent state."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from datetime import datetime, timezone
from uuid import uuid4


def now(): return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, state_root, artifact_root, manifest):
        self.root=Path(state_root);self.artifacts=Path(artifact_root);self.manifest=Path(manifest)
        self.lock=threading.RLock();self.process=None;self.thread=None;self.closing=threading.Event()
        self.root.mkdir(parents=True,exist_ok=True)
        for file in self.root.glob('*.json'):
            try:
                record=json.loads(file.read_text())
                if record['status'] in ('queued','running'):
                    record.update(status='interrupted',finished_at=now())
                    self._save(record)
            except (OSError,ValueError,KeyError): pass

    def _save(self,record):
        with self.lock:
            path=self.root/(record['id']+'.json');temp=path.with_suffix('.tmp')
            temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)

    def list(self):
        with self.lock:
            records=[json.loads(p.read_text()) for p in self.root.glob('*.json')]
        return sorted(records,key=lambda r:r['created_at'],reverse=True)

    def get(self,job_id):
        if not job_id.startswith('run-') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in job_id):
            raise FileNotFoundError('Job not found')
        with self.lock:
            path=self.root/(job_id+'.json')
            if not path.is_file(): raise FileNotFoundError('Job not found')
            record=json.loads(path.read_text())
            log=self.root/(job_id+'.log')
            record['log_tail']=log.read_text(errors='replace')[-12000:] if log.exists() else ''
        return record

    def start(self,request):
        with self.lock:
            if self.closing.is_set():
                raise RuntimeError('Training supervisor is shutting down')
            if self.thread and self.thread.is_alive():
                raise RuntimeError('Another training job is running')
            if not self.manifest.is_file(): raise FileNotFoundError('Dataset manifest is missing')
            job_id='run-'+uuid4().hex[:12]
            record=dict(id=job_id,status='queued',created_at=now(),request=request)
            self._save(record)
            self.thread=threading.Thread(target=self._run,args=(record,),daemon=True)
            self.thread.start()
            return record.copy()

    def _run(self,record):
        args=record['request'];output=self.artifacts/record['id']
        module='mammo_lens.deep' if args['kind']=='deep' else 'mammo_lens.cli'
        command=[sys.executable,'-m',module,str(self.manifest),'--output',str(output),'--mode',args['mode'],'--seed',str(args['seed'])]
        if args['kind']=='deep':
            command+=['--epochs',str(args['epochs']),'--batch-size',str(args['batch_size'])]
            for key,flag in [('density_aware','--density-aware'),('fine_tune','--fine-tune')]:
                if args[key]: command.append(flag)
        try:
            with (self.root/(record['id']+'.log')).open('w') as log:
                with self.lock:
                    if self.closing.is_set():
                        record.update(status='interrupted',finished_at=now());return
                    self.process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                record.update(status='running',started_at=now());self._save(record)
                code=self.process.wait()
            record.update(status='interrupted' if self.closing.is_set() else ('succeeded' if code==0 else 'failed'),exit_code=code,finished_at=now())
        except Exception as exc:
            record.update(status='failed',error=type(exc).__name__,finished_at=now())
        finally:
            self.process=None
            self._save(record)

    def close(self):
        self.closing.set()
        with self.lock:
            process=self.process
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill();process.wait()
        if self.thread: self.thread.join(timeout=12)
