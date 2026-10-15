from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1]
manifest=json.loads((R/'provenance/MANIFEST.json').read_text(encoding='utf-8'))
for row in manifest:
    path=R/row['path']
    assert path.is_file(),row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path']
print('PASS:',len(manifest),'payload files match SHA256 manifest')
