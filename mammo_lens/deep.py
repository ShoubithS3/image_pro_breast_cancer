"""Global/local ResNet18 with gated patch attention and optional density multitask head.

This implementation is an experimental composition of established methods, not a
claim of architectural novelty. All model selection/calibration uses validation data.
"""
import argparse
from copy import deepcopy
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from .core import MODES, features, load_manifest, metrics, patient_split, prepare_image
from .evaluation import calibrate_temperature, calibrated, evaluate_test


def image_tensors(pixels):
    _, boxes = features(pixels)
    def tensor(array, size):
        resized = Image.fromarray((array*255).astype(np.uint8)).resize((size,size))
        t = torch.from_numpy(np.array(resized, copy=True)).float().unsqueeze(0).repeat(3,1,1)/255
        mean = torch.tensor([.485,.456,.406])[:,None,None]
        std = torch.tensor([.229,.224,.225])[:,None,None]
        return (t-mean)/std
    return tensor(pixels,224), torch.stack([tensor(pixels[y:y1,x:x1],128) for x,y,x1,y1 in boxes]), boxes


class MammogramDataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows, self.augment = rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        pixels = prepare_image(row['image_path'])
        if self.augment and random.random() < .5:
            pixels = np.fliplr(pixels).copy()
        image, patches, _ = image_tensors(pixels)
        return image, patches, torch.tensor(row['label'],dtype=torch.float32), int(row['density'])-1 if row['density'] else -1


class MammoNet(nn.Module):
    def __init__(self, mode='combined', density_aware=False, pretrained=False, freeze_encoder=True):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Invalid feature mode')
        self.mode, self.density_aware = mode, density_aware
        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder.fc = nn.Identity()
        self.attention_v = nn.Linear(512,128)
        self.attention_u = nn.Linear(512,128)
        self.attention_w = nn.Linear(128,1)
        self.density_head = nn.Linear(512,4)
        dimension = 1024 if mode=='combined' else 512
        self.head = nn.Sequential(nn.Linear(dimension+(4 if density_aware else 0),128),nn.ReLU(),
                                  nn.Dropout(.25),nn.Linear(128,1))
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()  # Freeze BatchNorm running statistics too.
        return self

    def forward(self, image, patches):
        global_vector = self.encoder(image)
        if self.mode != 'global':
            batch,count,channels,height,width = patches.shape
            patch_vectors = self.encoder(patches.reshape(batch*count,channels,height,width)).reshape(batch,count,512)
            weights = self.attention_w(torch.tanh(self.attention_v(patch_vectors))*torch.sigmoid(self.attention_u(patch_vectors))).squeeze(-1).softmax(dim=1)
            local_vector = (patch_vectors*weights.unsqueeze(-1)).sum(dim=1)
        else:
            weights = image.new_zeros((image.shape[0],patches.shape[1]))
            local_vector = global_vector
        # Patch-only classification never consumes the whole-image feature vector.
        density_vector = local_vector if self.mode=='patch' else global_vector
        density_logits = self.density_head(density_vector)
        fused = torch.cat([global_vector,local_vector],dim=1) if self.mode=='combined' else (global_vector if self.mode=='global' else local_vector)
        if self.density_aware:
            fused = torch.cat([fused,density_logits.softmax(dim=1)],dim=1)
        return self.head(fused).squeeze(1), density_logits, weights


def device_for(name='auto'):
    if name=='auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return torch.device(name)


def infer_loader(model, loader, device):
    model.eval()
    scores, densities = [], []
    with torch.inference_mode():
        for images, patches, _, _ in loader:
            logits,density,_ = model(images.to(device),patches.to(device))
            scores.extend(logits.sigmoid().cpu().tolist())
            densities.extend(density.argmax(1).cpu().tolist())
    return np.array(scores), np.array(densities)


def load_deep(directory, device='cpu'):
    directory = Path(directory)
    config = json.loads((directory/'config.json').read_text())
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported checkpoint schema')
    model = MammoNet(mode=config['mode'],density_aware=config['density_aware'],pretrained=False)
    model.load_state_dict(torch.load(directory/'weights.pt',map_location='cpu',weights_only=True))
    return model.to(device).eval(),config


def predict_pixels(model, config, pixels, device='cpu'):
    image,patches,boxes = image_tensors(pixels)
    with torch.inference_mode():
        logits,density,attention = model(image[None].to(device),patches[None].to(device))
    score = float(calibrated(logits.sigmoid().cpu().numpy(),config.get('temperature',1.))[0])
    return dict(score=score, density_probabilities=density.softmax(1)[0].cpu().tolist() if config.get('density_trained') else None,
                patch_attention=attention[0].cpu().tolist() if config['mode']!='global' else None,boxes=boxes)


def train(manifest, output, mode='combined', epochs=10, batch_size=8, seed=42,
          pretrained=True, fine_tune=False, density_aware=False, device='auto', provenance='real'):
    if epochs<1 or batch_size<1:
        raise ValueError('Epochs and batch size must be positive')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    output=Path(output)
    if output.exists():
        raise ValueError('Choose a new output directory')
    rows=load_manifest(manifest)
    parts=patient_split(rows,seed)
    datasets={name:MammogramDataset([rows[i] for i in ids],augment=name=='train') for name,ids in parts.items()}
    generator=torch.Generator().manual_seed(seed)
    loaders={name:DataLoader(ds,batch_size=batch_size,shuffle=name=='train',num_workers=0,generator=generator)
             for name,ds in datasets.items()}
    density_trained=any(rows[i]['density'] for i in parts['train'])
    if density_aware and not density_trained:
        raise ValueError('Density-aware fusion requires density labels in the training partition')
    dev=device_for(device)
    model=MammoNet(mode,density_aware,pretrained,freeze_encoder=not fine_tune).to(dev)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4 if fine_tune else 1e-3,weight_decay=1e-4)
    best_loss=float('inf');best_state=None;history=[]
    validation_y=[r['label'] for r in datasets['validation'].rows]
    output.mkdir(parents=True)
    from sklearn.metrics import log_loss
    for epoch in range(epochs):
        model.train();total_loss=0
        for images,patches,labels,densities in loaders['train']:
            images,patches,labels,densities=images.to(dev),patches.to(dev),labels.to(dev),densities.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits,density_logits,_=model(images,patches)
            loss=F.binary_cross_entropy_with_logits(logits,labels)
            known=densities>=0
            if known.any():
                loss=loss+.2*F.cross_entropy(density_logits[known],densities[known])
            loss.backward();optimizer.step()
            total_loss+=float(loss.detach())*len(labels)
        validation_p,_=infer_loader(model,loaders['validation'],dev)
        val_loss=float(log_loss(validation_y,validation_p,labels=[0,1]))
        history.append(dict(epoch=epoch+1,training_loss=total_loss/len(datasets['train']),validation_log_loss=val_loss))
        print(json.dumps(history[-1]),flush=True)
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if val_loss<best_loss:
            best_loss=val_loss
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_epoch=epoch+1
    model.load_state_dict(best_state)
    validation_p,_=infer_loader(model,loaders['validation'],dev)
    temperature=calibrate_temperature(validation_y,validation_p)
    config=dict(schema_version=1,kind='deep',mode=mode,density_aware=density_aware,
                density_trained=density_trained,pretrained=pretrained,encoder='resnet18',
                freeze_encoder=not fine_tune,temperature=temperature,seed=seed,best_epoch=best_epoch,
                epochs=epochs,batch_size=batch_size,provenance=provenance,clinical_validation=False,
                preprocessing='percentile_1_99.5_gray_256_then_imagenet_normalization')
    torch.save(best_state,output/'weights.pt')
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    report=dict(config=config,splits={})
    for name,loader in loaders.items():
        # No random training augmentation during final evaluation.
        datasets[name].augment=False
        eval_loader=DataLoader(datasets[name],batch_size=batch_size,shuffle=False)
        raw,density_p=infer_loader(model,eval_loader,dev)
        scores=calibrated(raw,temperature)
        labels=[r['label'] for r in datasets[name].rows]
        report['splits'][name]=dict(metrics=metrics(labels,scores),uncalibrated_metrics=metrics(labels,raw),
                                   patient_count=len({r['patient_id'] for r in datasets[name].rows}))
        if name=='test':
            report['held_out']=evaluate_test(datasets[name].rows,scores,seed)
            known=np.array([bool(r['density']) for r in datasets[name].rows])
            truth=np.array([int(r['density'])-1 if r['density'] else -1 for r in datasets[name].rows])
            report['density_assessment']=dict(n=int(known.sum()),accuracy=float((truth[known]==density_p[known]).mean()) if known.any() and density_trained else None)
    test_rows=datasets['test'].rows
    report['stress_test']={}
    for name,kwargs in {'blur_radius_2':{'blur':2},'noise_std_0.05':{'noise':.05}}.items():
        scores=[predict_pixels(model,config,prepare_image(r['image_path'],seed=seed+i,**kwargs),dev)['score'] for i,r in enumerate(test_rows)]
        report['stress_test'][name]=metrics([r['label'] for r in test_rows],scores)
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    assignments={int(i):name for name,ids in parts.items() for i in ids}
    (output/'split_manifest.json').write_text(json.dumps([dict(r,split=assignments[i]) for i,r in enumerate(rows)],indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest');parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=MODES,default='combined')
    parser.add_argument('--epochs',type=int,default=10);parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--no-pretrained',action='store_true');parser.add_argument('--fine-tune',action='store_true')
    parser.add_argument('--density-aware',action='store_true')
    parser.add_argument('--provenance',choices=['real','synthetic-smoke'],default='real')
    args=parser.parse_args()
    report=train(args.manifest,args.output,args.mode,args.epochs,args.batch_size,args.seed,
                 not args.no_pretrained,args.fine_tune,args.density_aware,args.device,args.provenance)
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':
    main()
