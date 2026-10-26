from io import BytesIO
import time

import numpy as np
from fastapi.testclient import TestClient
import pytest

from mammo_lens.api import create_app
from mammo_lens.cli import train
from mammo_lens.core import load_manifest


def test_image_api_handles_real_upload_and_invalid_requests(tmp_path, cohort, monkeypatch):
    monkeypatch.delenv('MAMMO_ENABLE_TRAINING',raising=False)
    with TestClient(create_app(tmp_path/'artifacts',tmp_path/'state',cohort)) as client:
        assert client.get('/').status_code==200
        assert client.get('/api/health').json()['dataset_available'] is True
        assert client.get('/api/models').json()==[]
        raw=open(load_manifest(cohort)[0]['image_path'],'rb').read()
        result=client.post('/api/analyze',files={'image':('image.png',raw,'image/png')},data={'noise':'.05'})
        assert result.status_code==200,result.text
        body=result.json()
        assert body['prediction'] is None
        assert len(body['patches'])==4
        assert body['measurements']['mean_absolute_pixel_change']>0
        assert body['reference_image'].startswith('data:image/png;base64,')
        assert client.post('/api/analyze',files={'image':('bad.png',b'not an image')}).status_code==422
        assert client.post('/api/analyze',files={'image':('image.png',raw)},data={'noise':1}).status_code==422
        assert client.post('/api/analyze',files={'image':('image.png',raw)},data={'model_id':'../../outside'}).status_code==404
        assert not (tmp_path/'state').exists()  # No uploaded image or job state retained.
        monkeypatch.setattr('mammo_lens.api.MAX_UPLOAD',5)
        assert client.post('/api/analyze',files={'image':('image.png',raw)}).status_code==413


def test_model_registry_prediction_and_masking(tmp_path,cohort):
    root=tmp_path/'artifacts'
    train(cohort,root/'baseline')
    with TestClient(create_app(root,tmp_path/'state',cohort)) as client:
        assert client.get('/api/models').json()[0]['mode']=='combined'
        raw=open(load_manifest(cohort)[0]['image_path'],'rb').read()
        result=client.post('/api/analyze',files={'image':('image.png',raw)},data={'model_id':'baseline','blur':2})
        assert result.status_code==200,result.text
        p=result.json()['prediction']
        assert 0<=p['score']<=1
        assert len(p['patch_masking'])==4
        assert p['score_shift']==pytest.approx(p['score']-p['reference_score'])
        assert p['threshold_validated'] is False
        assert 'patient_id' not in client.get('/api/experiments').text


def test_authenticated_training_job_runs_and_persists(tmp_path,cohort,monkeypatch):
    monkeypatch.setenv('MAMMO_ENABLE_TRAINING','1')
    monkeypatch.setenv('MAMMO_ADMIN_TOKEN','test-only-operator-token')
    headers={'Authorization':'Bearer test-only-operator-token'}
    app=create_app(tmp_path/'artifacts',tmp_path/'state',cohort)
    with TestClient(app) as client:
        assert client.get('/api/dataset').status_code==401
        assert client.get('/api/dataset',headers=headers).json()['patients']==60
        job=client.post('/api/jobs',headers=headers,json={'kind':'baseline','mode':'global'}).json()
        assert job['id'].startswith('run-')
        for _ in range(300):
            record=client.get('/api/jobs/'+job['id'],headers=headers).json()
            if record['status'] in ('failed','succeeded'): break
            time.sleep(.05)
        assert record['status']=='succeeded',record
        assert len(client.get('/api/models').json())==1
    assert (tmp_path/'state'/(job['id']+'.json')).is_file()


def test_training_requires_explicit_enable_and_token(tmp_path,monkeypatch):
    monkeypatch.setenv('MAMMO_ENABLE_TRAINING','1')
    monkeypatch.delenv('MAMMO_ADMIN_TOKEN',raising=False)
    with pytest.raises(RuntimeError,match='requires MAMMO_ADMIN_TOKEN'):
        with TestClient(create_app(tmp_path/'artifacts',tmp_path/'state')): pass
