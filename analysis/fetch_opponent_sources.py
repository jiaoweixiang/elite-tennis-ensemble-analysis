"""Optionally retrieve pinned upstream match CSVs without redistributing them."""
from pathlib import Path
import argparse,hashlib,json,urllib.request

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/third_party/opponent_sources')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=json.loads((ROOT/'data/reference/revision2026/opponent/sources/download_manifest.json').read_text(encoding='utf-8'))
    for record in records:
        name=record['file'].removeprefix('source:')
        if not (name.startswith('atp_matches_') and name.endswith('.csv')):continue
        target=args.output/name
        if target.exists():data=target.read_bytes()
        else:
            request=urllib.request.Request(record['pinned_url'],headers={'User-Agent':'elite-tennis-ensemble-analysis/1.0.2'})
            with urllib.request.urlopen(request,timeout=90) as response:data=response.read()
        if hashlib.sha256(data).hexdigest()!=record['sha256']:raise RuntimeError('SHA-256 mismatch: '+name)
        target.write_bytes(data)
        print('Verified',name,flush=True)
    print('Upstream CC BY-NC-SA 4.0 continues to apply; see LICENSE and data/README.md.')
if __name__=='__main__':main()
