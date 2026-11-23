"""Refresh integrity manifests after any change to deposited files."""
from pathlib import Path
import csv,hashlib

ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write(path,files):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','sha256','bytes'],lineterminator='\n')
        writer.writeheader()
        for item in files:writer.writerow(dict(path=item.relative_to(ROOT).as_posix(),sha256=digest(item),bytes=item.stat().st_size))

def main():
    data=[ROOT/'data/indicator_dictionary.csv']
    for folder in ['raw','processed','reference']:
        data+=sorted(p for p in (ROOT/'data'/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    write(ROOT/'data/sha256.csv',data)
    manifest=ROOT/'MANIFEST_SHA256.csv'
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p!=manifest
        and p.relative_to(ROOT).parts[0] not in ('outputs','.git','.venv')
        and '__pycache__' not in p.parts and p.suffix!='.pyc')
    write(manifest,files)
    print('Updated data/sha256.csv and MANIFEST_SHA256.csv. Rerun python run_all.py before deposition.')

if __name__=='__main__':main()
