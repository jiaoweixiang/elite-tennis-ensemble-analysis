"""Verify the Original column of recoding records against the modelling inputs."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    raw=[]
    for n in ['train_grouped.csv','test_grouped.csv']:
        with (ROOT/'data/processed'/n).open(encoding='utf-8-sig') as f:raw+=list(csv.DictReader(f))
    lookup={(r['match_id'],int(r['set_within_match'])):r for r in raw}
    with (ROOT/'data/reference/revision2026/reliability/observer_records.csv').open(encoding='utf-8-sig') as f:obs=list(csv.DictReader(f))
    checked=0
    for row in obs:
        if row['indicator'].startswith('Set outcome'):continue
        source=lookup[(row['match_id'],int(row['set_number']))];keys={k.strip():k for k in source}
        assert abs(float(source[keys[row['indicator'].strip()]])-float(row['original']))<1e-8,(row['match_id'],row['set_number'],row['indicator'])
        checked+=1
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    (out/'recoding_origin_check.json').write_text(json.dumps(dict(status='PASS',original_values_checked=checked,model_inputs_changed=False),indent=2),encoding='utf-8')
    print('Verified',checked,'original recoding-sample values against modelling inputs.',flush=True)
if __name__=='__main__':main()

