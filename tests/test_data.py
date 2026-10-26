import csv
import shutil
from pathlib import Path

from mammo_lens.core import load_manifest, patient_split
from mammo_lens.data import build_cbis_manifest


def write_csv(path,fields,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def test_cbis_uid_mapping_aggregation_and_official_patient_exclusion(cohort,tmp_path):
    root=tmp_path/'cbis';(root/'csv').mkdir(parents=True)
    original=load_manifest(cohort)
    info=[];train=[];test=[]
    for index,row in enumerate(original):
        uid=f'uid-{index}';path=root/'jpeg'/uid/'1.jpg';path.parent.mkdir(parents=True)
        shutil.copyfile(row['image_path'],path)
        info.append(dict(SeriesDescription='full mammogram images',image_path=f'CBIS-DDSM/jpeg/{uid}/1.jpg'))
        record={'patient_id':row['patient_id'],'image file path':f'Mass-Training/{uid}/image.dcm',
                'breast_density':row['density'],'pathology':'MALIGNANT' if row['label'] else 'BENIGN'}
        if index//2>=48 and index%2==0: test.append(record)
        else: train.append(record)
    fields=list(train[0])
    write_csv(root/'csv/dicom_info.csv',['SeriesDescription','image_path'],info)
    write_csv(root/'csv/mass_case_description_train_set.csv',fields,train)
    write_csv(root/'csv/mass_case_description_test_set.csv',fields,test)
    # Same full image with another malignant lesion: aggregate, never duplicate rows.
    write_csv(root/'csv/calc_case_description_train_set.csv',fields,[dict(train[0],pathology='MALIGNANT')])
    write_csv(root/'csv/calc_case_description_test_set.csv',fields,[])
    out=tmp_path/'manifest.csv'
    report=build_cbis_manifest(root,out)
    assert report['excluded_train_views_for_test_patients']==12
    rows=load_manifest(out);parts=patient_split(rows)
    assert len(rows)==108
    assert rows[0]['label']==1
    test_patients={rows[i]['patient_id'] for i in parts['test']}
    train_patients={rows[i]['patient_id'] for i in parts['train']}
    assert not test_patients & train_patients
    assert (out.with_suffix('.audit.json')).is_file()
