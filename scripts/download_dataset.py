#!/usr/bin/env python3
"""Download the assignment's public Kaggle export with verified TLS and safe extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import urllib.error
import urllib.parse
import urllib.request
import zipfile

DATASET='awsaf49/cbis-ddsm-breast-cancer-image-dataset'


def download(destination):
    destination=Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Destination is not empty; keep existing data and choose a new directory')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination.parent).free<15*1024**3:
        raise ValueError('At least 15 GiB free disk space is required for archive and extraction')
    archive=destination.parent/'cbis-ddsm.zip'
    if archive.exists(): raise ValueError('Archive already exists; verify/extract it or choose another directory')
    temporary=archive.with_suffix('.zip.part')
    digest=hashlib.sha256();count=0;next_report=64*1024**2
    try:
        request=urllib.request.Request('https://www.kaggle.com/api/v1/datasets/download/'+DATASET,
                                       headers={'User-Agent':'MAMMO-LENS/research-dataset-download'})
        with urllib.request.urlopen(request,timeout=60) as response, temporary.open('wb') as handle:
            while chunk:=response.read(1024**2):
                count+=len(chunk)
                if count>10*1024**3: raise ValueError('Unexpected archive size; stopping at 10 GiB')
                handle.write(chunk);digest.update(chunk)
                if count>=next_report:
                    print(f'Downloaded {count/1024**2:.0f} MiB',flush=True);next_report+=64*1024**2
        temporary.rename(archive)
        with zipfile.ZipFile(archive) as z:
            total=sum(item.file_size for item in z.infolist())
            if total>20*1024**3 or shutil.disk_usage(destination.parent).free<total+1024**3:
                raise ValueError('Insufficient disk space or unexpected uncompressed dataset size')
            for item in z.infolist():
                target=(destination/item.filename).resolve()
                if not target.is_relative_to(destination) or stat.S_ISLNK(item.external_attr>>16):
                    raise ValueError('Unsafe archive member rejected')
            # zipfile checks CRC during extraction; TLS remains enabled throughout download.
            destination.mkdir(parents=True,exist_ok=True)
            z.extractall(destination)
        receipt=dict(dataset=DATASET,archive_bytes=count,archive_sha256=digest.hexdigest(),
                     hash_is_local_receipt_not_publisher_signature=True)
        (destination/'download-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('Dataset extracted. Receipt saved. Archive retained; remove manually after validating the manifest.')
    except urllib.error.HTTPError as exc:
        # Signed redirect URLs can contain credentials; never print them.
        raise RuntimeError(f'Download HTTP {exc.code} from {urllib.parse.urlsplit(exc.url).hostname}. Check environment network access.') from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Network request failed ({type(exc.reason).__name__}). Required hosts: www.kaggle.com, storage.googleapis.com. No credentials were requested.') from None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',default='data/cbis-ddsm')
    args=parser.parse_args()
    try: download(args.destination)
    except (ValueError,RuntimeError,OSError,zipfile.BadZipFile) as exc:
        parser.exit(1,f'{exc}\n')

if __name__=='__main__': main()
