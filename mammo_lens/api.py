"""Same-origin research API and frontend. Images are never retained by this service."""
from contextlib import asynccontextmanager
from io import BytesIO
import base64
import json
import os
from pathlib import Path
import secrets
import threading
from typing import Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .core import features, prepare_image, load_manifest, patient_split
from .inference import ModelRegistry
from .jobs import JobManager

PROJECT=Path(__file__).resolve().parents[1]
MAX_UPLOAD=20*1024*1024


def encode_image(pixels, boxes=None):
    image=Image.fromarray((pixels*255).astype(np.uint8)).convert('RGB')
    if boxes:
        draw=ImageDraw.Draw(image)
        for i,(x,y,x1,y1) in enumerate(boxes,1):
            draw.rectangle((x,y,x1-1,y1-1),outline='#d4f5a0',width=1)
            draw.text((x+3,y+3),f'P{i}',fill='#d4f5a0')
    buffer=BytesIO();image.save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


class TrainingRequest(BaseModel):
    kind:Literal['baseline','deep']='deep'
    mode:Literal['global','patch','combined']='combined'
    epochs:int=Field(default=10,ge=1,le=50)
    batch_size:int=Field(default=8,ge=2,le=32)
    seed:int=Field(default=42,ge=0,le=2**31-1)
    density_aware:bool=False
    fine_tune:bool=False


def create_app(artifact_root=None, state_root=None, manifest=None):
    root=Path(artifact_root or os.environ.get('MAMMO_ARTIFACTS',PROJECT/'artifacts')).resolve()
    state=Path(state_root or os.environ.get('MAMMO_STATE',PROJECT/'state')).resolve()
    manifest=Path(manifest or os.environ.get('MAMMO_MANIFEST',PROJECT/'data/manifest.csv')).resolve()
    registry=ModelRegistry(root)
    guard=threading.Lock()
    manager=None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal manager
        if os.environ.get('MAMMO_ENABLE_TRAINING')=='1':
            if not os.environ.get('MAMMO_ADMIN_TOKEN'):
                raise RuntimeError('Training requires MAMMO_ADMIN_TOKEN; use the CLI if you do not need web training')
            manager=JobManager(state,root,manifest)
        yield
        if manager: manager.close()

    app=FastAPI(title='MAMMO-LENS research API',version='1.0.0',lifespan=lifespan)
    app.state.registry=registry

    @app.middleware('http')
    async def security_headers(request,call_next):
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['X-Frame-Options']='DENY'
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control']='no-store'
        return response

    def admin(authorization:str|None=Header(default=None)):
        token=os.environ.get('MAMMO_ADMIN_TOKEN')
        if not token or not authorization or not secrets.compare_digest(authorization,'Bearer '+token):
            raise HTTPException(401,'Operator authentication required')
        if manager is None:
            raise HTTPException(403,'Web training is disabled; use the CLI or explicitly enable operator mode')

    @app.get('/api/health')
    def health():
        return dict(status='ok',models=len(registry.list()),dataset_available=manifest.is_file(),
                    web_training_enabled=manager is not None,clinical_validation=False)

    @app.get('/api/models')
    def models(): return registry.list()

    @app.get('/api/experiments')
    def experiments():
        result=[]
        for row in registry.list():
            report=json.loads((registry.directory(row['id'])/'metrics.json').read_text())
            result.append(dict(**row,report=report))
        return result

    def analyze_bytes(raw,model_id,blur,noise,mask_patches):
        try:
            with Image.open(BytesIO(raw)) as image:
                if image.format not in ('PNG','JPEG','TIFF'):
                    raise ValueError('Supported formats: PNG, JPEG, TIFF')
                if image.width*image.height>40_000_000:
                    raise ValueError('Image exceeds the 40-megapixel limit; export a smaller copy')
                image.verify()
            clean=prepare_image(BytesIO(raw))
            pixels=prepare_image(BytesIO(raw),blur=blur,noise=noise)
        except (ValueError,OSError,UnidentifiedImageError,Image.DecompressionBombError) as exc:
            raise HTTPException(422,str(exc)) from exc
        _,boxes=features(pixels)
        patch_info=[dict(id=f'P{i+1}',box=box,texture_std=float(pixels[y:y1,x:x1].std())) for i,box in enumerate(boxes) for x,y,x1,y1 in [box]]
        prediction=None
        if model_id:
            try:
                with guard:
                    original,config=registry.predict(model_id,clean)
                    current,_=registry.predict(model_id,pixels)
                    effects=[]
                    if mask_patches:
                        for box in boxes:
                            masked=pixels.copy();x,y,x1,y1=box
                            masked[y:y1,x:x1]=np.median(pixels)
                            changed,_=registry.predict(model_id,masked)
                            effects.append(dict(box=box,score=changed['score'],delta=changed['score']-current['score']))
                prediction=dict(model_id=model_id,kind=config['kind'],mode=config['mode'],provenance=config.get('provenance','unverified'),
                                reference_score=original['score'],score=current['score'],score_shift=current['score']-original['score'],
                                low_confidence=max(current['score'],1-current['score'])<.7,
                                confidence_threshold=.7,threshold_validated=False,patch_masking=effects,
                                density_probabilities=current['density_probabilities'],patch_attention=current['patch_attention'])
            except FileNotFoundError as exc: raise HTTPException(404,str(exc)) from exc
            except ValueError as exc: raise HTTPException(422,str(exc)) from exc
        return dict(reference_image=encode_image(clean),processed_image=encode_image(pixels,boxes),patches=patch_info,
                    measurements=dict(mean_absolute_pixel_change=float(np.abs(pixels-clean).mean()),intensity_std=float(pixels.std())),
                    preprocessing='percentile_1_99.5_gray_256',transformations=dict(blur_radius=blur,noise_std=noise,noise_seed=42),
                    prediction=prediction,trained_inference_performed=prediction is not None,clinical_validation=False,
                    limitations=['Texture patches are not lesion detections.','Masking recomputes patch selection and measures a full-pipeline perturbation, not causal or clinical evidence.',
                                 'Scores are experimental; not future risk or validated clinical probabilities.'])

    @app.post('/api/analyze')
    async def analyze(image:UploadFile=File(...),model_id:str=Form(default=''),blur:float=Form(default=0,ge=0,le=5),
                      noise:float=Form(default=0,ge=0,le=.2),mask_patches:bool=Form(default=True)):
        raw=await image.read(MAX_UPLOAD+1)
        await image.close()
        if len(raw)>MAX_UPLOAD: raise HTTPException(413,'Image exceeds 20 MB')
        return await run_in_threadpool(analyze_bytes,raw,model_id,blur,noise,mask_patches)

    @app.get('/api/dataset',dependencies=[Depends(admin)])
    def dataset_audit():
        try:
            rows=load_manifest(manifest);parts=patient_split(rows)
        except (ValueError,OSError) as exc: raise HTTPException(422,str(exc)) from exc
        return dict(images=len(rows),patients=len({r['patient_id'] for r in rows}),
                    class_counts={str(k):sum(r['label']==k for r in rows) for k in (0,1)},
                    density_available=sum(bool(r['density']) for r in rows),splits={k:len(v) for k,v in parts.items()})

    @app.post('/api/jobs',status_code=202,dependencies=[Depends(admin)])
    def start_job(request:TrainingRequest):
        try: return manager.start(request.model_dump())
        except FileNotFoundError as exc: raise HTTPException(422,str(exc)) from exc
        except RuntimeError as exc: raise HTTPException(409,str(exc)) from exc

    @app.get('/api/jobs',dependencies=[Depends(admin)])
    def jobs(): return manager.list()

    @app.get('/api/jobs/{job_id}',dependencies=[Depends(admin)])
    def job(job_id:str):
        try: return manager.get(job_id)
        except FileNotFoundError as exc: raise HTTPException(404,str(exc)) from exc

    frontend=PROJECT/'frontend'
    if frontend.is_dir():
        app.mount('/assets',StaticFiles(directory=frontend),name='frontend')
        @app.get('/',include_in_schema=False)
        def index(): return FileResponse(frontend/'index.html')
    return app

app=create_app()
