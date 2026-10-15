"""Locally convert official K-Radar v2.1 annotations; does not download or upload."""
from pathlib import Path
import argparse,csv,io,json,math,re,zipfile,hashlib
from decimal import Decimal
R=Path(__file__).resolve().parents[1]
EXPECTED_SHA='8d8c1f3033ded1825ec740b4df45d4da2b505f5b489b902cfd4a723124788a02'
PREFIX='kradar_revised_label_v2_1/KRadar_revised_visibility/1/'
def main():
    p=argparse.ArgumentParser();p.add_argument('--labels-zip',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    assert not a.out.exists(),'Use a new output filename'
    digest=hashlib.sha256(a.labels_zip.read_bytes()).hexdigest()
    assert digest==EXPECTED_SHA,'The official annotation release differs from the recorded version'
    with (R/'data/target/prepared/frame_manifest.csv').open(encoding='utf-8-sig',newline='') as f:frames=list(csv.DictReader(f))
    records=[]
    with zipfile.ZipFile(a.labels_zip) as z:
        for frame in frames:
            name=PREFIX+frame['label_filename'];body=z.read(name).decode('utf-8').strip()
            m=re.match(r'\* idx\([^)]*\)=([0-9_]+),\s*timestamp=([0-9.]+)',body);assert m
            assert int(Decimal(m.group(2))*1000000000)==int(frame['timestamp_ns'])
            for row in csv.reader(io.StringIO(body[m.end():].strip()),skipinitialspace=True):
                if not row:continue
                assert len(row)==11 and row[0]=='*'
                x,y,zc,angle,hl,hw,hh=map(float,row[4:])
                records.append(dict(label=row[3],object_id=row[2],visibility=row[1],center_x=x,center_y=y,center_z=zc,size_x=2*hl,size_y=2*hw,size_z=2*hh,yaw=math.radians(angle),timestamp=m.group(2),frame_id='kradar_os2_64',source_path='official_KRadar_v2_1::'+name,sequence='1',radar_index=frame['radar_index']))
    assert len(frames)==597 and len(records)==1890
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(json.dumps({'frames':597,'GT_boxes':1890,'official_zip_sha256':digest,'output':'local converted CSV','no_GT_tuning':True},indent=2))
if __name__=='__main__':main()
