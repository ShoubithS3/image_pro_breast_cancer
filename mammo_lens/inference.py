"""Trusted local artifact registry and shared inference for the API."""
import json
import re
from pathlib import Path

import joblib
import numpy as np

from .core import features
from .evaluation import calibrated


class ModelRegistry:
    def __init__(self, root):
        self.root=Path(root).resolve()
        self.cache={}

    def directory(self, model_id):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',model_id):
            raise ValueError('Invalid model ID')
        path=(self.root/model_id).resolve()
        if not path.is_relative_to(self.root) or not path.is_dir():
            raise FileNotFoundError('Model not found')
        return path

    def list(self):
        result=[]
        if not self.root.exists(): return result
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',path.name): continue
            try:
                path=self.directory(path.name)
                if not (path/'metrics.json').is_file(): continue  # Never expose a partly trained run.
                config=json.loads((path/'config.json').read_text()) if (path/'config.json').exists() else dict(kind='baseline',mode=path.name,provenance='unverified')
                if not ((path/'weights.pt').exists() or (path/'model.joblib').exists()): continue
                result.append(dict(id=path.name,kind=config['kind'],mode=config['mode'],
                                   provenance=config.get('provenance','unverified'),density_trained=config.get('density_trained',False)))
            except (OSError,ValueError,KeyError):
                continue
        return result

    def load(self, model_id):
        if model_id not in {r['id'] for r in self.list()}:
            raise FileNotFoundError('Completed model not found')
        directory=self.directory(model_id)
        stamp=(directory/'metrics.json').stat().st_mtime_ns
        if model_id not in self.cache or self.cache[model_id][0]!=stamp:
            if (directory/'config.json').is_file():
                config=json.loads((directory/'config.json').read_text())
            else:
                config=dict(kind='baseline',provenance='unverified')
            if config['kind']=='deep':
                from .deep import load_deep
                import torch
                torch.set_num_threads(2)
                model,config=load_deep(directory)
            else:
                # Only operator-created local artifacts; no model uploads are accepted.
                model=joblib.load(directory/'model.joblib')
                config.update(mode=model['mode'],temperature=model.get('temperature',1.))
            self.cache[model_id]=(stamp,model,config)
        _,model,config=self.cache[model_id]
        return model,config

    def predict(self, model_id, pixels):
        model,config=self.load(model_id)
        if config.get('provenance')=='synthetic-smoke':
            raise ValueError('Synthetic smoke-test artifacts cannot be used for disease scoring')
        if config['kind']=='deep':
            from .deep import predict_pixels
            result=predict_pixels(model,config,pixels)
        else:
            score=model['model'].predict_proba([features(pixels,config['mode'])[0]])[0,1]
            result=dict(score=float(calibrated([score],config.get('temperature',1.))[0]),
                        patch_attention=None,density_probabilities=None)
        return result,config
