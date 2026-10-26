"""Audit and convert the Kaggle CBIS-DDSM JPEG export without guessing labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from .core import load_manifest, patient_split


def build_cbis_manifest(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Refusing to overwrite an existing manifest')
    info = root / 'csv/dicom_info.csv'
    if not info.is_file():
        raise ValueError('Expected csv/dicom_info.csv below the dataset root')
    full = {}
    with info.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('SeriesDescription', '').strip().lower() != 'full mammogram images':
                continue
            raw = row['image_path'].replace('\\', '/').strip()
            if '/jpeg/' not in '/' + raw:
                raise ValueError('Unrecognized dicom_info image path; audit the export')
            relative = 'jpeg/' + ('/'+raw).split('/jpeg/', 1)[-1]
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Image path escapes the dataset root')
            full.setdefault(path.parent.name, set()).add(path)
    images, patients_test = {}, set()
    report = dict(source='awsaf49/cbis-ddsm-breast-cancer-image-dataset',
                  label_rule='full image is malignant if any described lesion is malignant',
                  benign_rule='BENIGN and BENIGN_WITHOUT_CALLBACK map to 0',
                  skipped_missing_files=0, excluded_train_views_for_test_patients=0)
    for kind in ('mass', 'calc'):
        for split in ('train', 'test'):
            source = root / f'csv/{kind}_case_description_{split}_set.csv'
            with source.open(newline='') as handle:
                for row in csv.DictReader(handle):
                    raw = row['image file path'].strip().replace('\\', '/')
                    candidates = set()
                    for component in raw.split('/')[:-1]:
                        candidates.update(full.get(component, set()))
                    if len(candidates) != 1:
                        raise ValueError(f'Ambiguous or unmapped full image for patient {row["patient_id"]}')
                    path = candidates.pop()
                    if not path.is_file():
                        report['skipped_missing_files'] += 1
                        continue
                    label = row['pathology'].strip()
                    if label not in ('MALIGNANT', 'BENIGN', 'BENIGN_WITHOUT_CALLBACK'):
                        raise ValueError(f'Unknown pathology category: {label}')
                    patient = row['patient_id'].strip()
                    density = (row.get('breast_density') or row.get('breast density') or '').strip()
                    if density in ('0', '0.0'):
                        density = ''
                    record = dict(patient_id=patient, image_path=str(path), label=int(label == 'MALIGNANT'),
                                  density=density, source_split=split)
                    if path in images:
                        old = images[path]
                        if old['patient_id'] != patient or old['density'] != density:
                            raise ValueError('Conflicting patient or density metadata for the same image')
                        record['label'] = max(record['label'], old['label'])
                        if old['source_split'] == 'test':
                            record['source_split'] = 'test'
                    images[path] = record
                    if split == 'test':
                        patients_test.add(patient)
    records = []
    for row in images.values():
        if row['source_split'] == 'train' and row['patient_id'] in patients_test:
            report['excluded_train_views_for_test_patients'] += 1
            continue
        records.append(row)
    if not records:
        raise ValueError('No usable images found')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate before publishing a manifest; retain exact absolute paths for local use.
    temp = output.with_suffix('.pending.csv')
    try:
        with temp.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        validated = load_manifest(temp)
        parts = patient_split(validated)
        report.update(image_count=len(records), patient_count=len({r['patient_id'] for r in records}),
                      split_images={k:len(v) for k,v in parts.items()})
        temp.rename(output)
    finally:
        temp.unlink(missing_ok=True)
    output.with_suffix('.audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--output', default='data/manifest.csv')
    args = parser.parse_args()
    print(json.dumps(build_cbis_manifest(args.root, args.output), indent=2))

if __name__ == '__main__':
    main()
