#!/usr/bin/env python3
"""Exercise the training stack on synthetic texture. Never a disease benchmark."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/synthetic-smoke')
    parser.add_argument('--output',default='artifacts/synthetic-smoke')
    parser.add_argument('--deep',action='store_true')
    args=parser.parse_args()
    root=Path(args.root)
    if root.exists(): parser.error('Use a new synthetic data directory to preserve previous outputs')
    root.mkdir(parents=True)
    rng=np.random.default_rng(7)
    manifest=root/'manifest.csv'
    with manifest.open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['patient_id','image_path','label','density'])
        for patient in range(40):
            for view in range(2):
                path=root/f'{patient}_{view}.png'
                # Random texture with arbitrary labels; no disease simulation or label signal.
                Image.fromarray(rng.integers(0,256,(80,80),dtype=np.uint8)).save(path)
                writer.writerow([f'synthetic_{patient}',path.name,patient%2,patient%4+1])
    if args.deep:
        from mammo_lens.deep import train
        train(manifest,args.output,epochs=1,batch_size=4,pretrained=False,device='cpu',provenance='synthetic-smoke')
    else:
        from mammo_lens.cli import train
        train(manifest,args.output,provenance='synthetic-smoke')
    print('SYNTHETIC SOFTWARE SMOKE TEST ONLY. Do not report these scores as medical results.')

if __name__=='__main__': main()
