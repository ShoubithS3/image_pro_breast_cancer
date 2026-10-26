import csv
import json

import numpy as np
import pytest
import torch

from mammo_lens.deep import MammoNet, load_deep, predict_pixels, train
from mammo_lens.inference import ModelRegistry
from mammo_lens.core import load_manifest, patient_split
from mammo_lens.evaluation import calibrated, calibrate_temperature, evaluate_test


def test_attention_and_density_head_receive_gradients():
    torch.set_num_threads(2)
    model=MammoNet(mode='combined',density_aware=True,pretrained=False)
    model.train()
    logits,density,attention=model(torch.rand(2,3,224,224),torch.rand(2,4,3,128,128))
    loss=logits.square().mean()+density.square().mean()
    loss.backward()
    assert torch.allclose(attention.sum(1),torch.ones(2))
    assert model.attention_w.weight.grad.abs().sum()>0
    assert model.density_head.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.encoder.parameters())
    assert not model.encoder.training


def test_deep_training_checkpoint_calibration_and_reload(cohort,tmp_path):
    with cohort.open() as handle: rows=list(csv.DictReader(handle))[:40]
    small=cohort.with_name('small.csv')
    with small.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    output=tmp_path/'artifacts'/'smoke'
    report=train(small,output,epochs=1,batch_size=4,pretrained=False,density_aware=True,device='cpu',provenance='synthetic-smoke')
    assert report['config']['provenance']=='synthetic-smoke'
    assert report['config']['density_trained'] is True
    assert report['held_out']['patient_cluster_95ci']['brier']['valid_resamples']==200
    model,config=load_deep(output)
    result=predict_pixels(model,config,np.random.default_rng(2).random((256,256)).astype(np.float32))
    assert 0<=result['score']<=1
    assert len(result['patch_attention'])==4
    assert sum(result['density_probabilities'])==pytest.approx(1)
    registry=ModelRegistry(output.parent)
    with pytest.raises(ValueError,match='Synthetic'):
        registry.predict('smoke',np.zeros((256,256)))


def test_official_split_rejects_overlapping_patients(cohort):
    rows=load_manifest(cohort)
    for i,row in enumerate(rows): row['source_split']='train' if i%2 else 'test'
    with pytest.raises(ValueError,match='overlap patients'):
        patient_split(rows)


def test_temperature_fits_only_supplied_validation_scores():
    labels=[0,1,0,1]
    scores=[.4,.6,.9,.1]
    temperature=calibrate_temperature(labels,scores)
    assert .1<=temperature<=10
    assert np.isfinite(calibrated(scores,temperature)).all()
