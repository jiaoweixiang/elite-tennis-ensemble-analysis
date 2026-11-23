"""Diagnose original ensemble probabilities; no probability updating is performed."""
from pathlib import Path
import csv,math,json
ROOT=Path(__file__).resolve().parents[1]
def expit(x):
    return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))
def loss(a,b,x,y):
    return sum(max(z:=a+b*t,0)+math.log1p(math.exp(-abs(z)))-s*z for t,s in zip(x,y))
def diag(y,p):
    x=[math.log(v/(1-v)) for v in p]
    a,b=0.,1.
    for _ in range(100):
        q=[expit(a+b*t) for t in x]
        g0=sum(u-s for u,s in zip(q,y));g1=sum((u-s)*t for u,s,t in zip(q,y,x))
        if max(abs(g0),abs(g1))<1e-10:break
        w=[u*(1-u) for u in q]
        h00=sum(w);h01=sum(u*t for u,t in zip(w,x));h11=sum(u*t*t for u,t in zip(w,x))
        det=h00*h11-h01*h01
        assert det>0
        da=(h11*g0-h01*g1)/det;db=(-h01*g0+h00*g1)/det
        step=1.;old=loss(a,b,x,y)
        while loss(a-step*da,b-step*db,x,y)>old+1e-12:
            step/=2
            assert step>1e-12
        a-=step*da;b-=step*db
    assert max(abs(g0),abs(g1))<1e-7
    lo,hi=-100.,100.
    for _ in range(100):
        mid=(lo+hi)/2
        if sum(expit(t+mid)-s for t,s in zip(x,y))>0:hi=mid
        else:lo=mid
    return dict(joint_calibration_intercept=a,calibration_slope=b,
                calibration_in_the_large=(lo+hi)/2,
                brier=sum((s-t)**2 for s,t in zip(y,p))/len(y))

def main():
    rows=[]
    for sample,name in [('development_outer','outer_predictions.csv'),('temporal_2025','test_predictions.csv')]:
        with (ROOT/'data/reference'/name).open(encoding='utf-8-sig') as f:frame=list(csv.DictReader(f))
        y=[int(r['label']) for r in frame];p=[float(r['auc_selector_score']) for r in frame]
        result=diag(y,p)
        rows.append(dict(sample=sample,n=len(y),wins=sum(y),calibration_intercept=result['calibration_in_the_large'],calibration_slope=result['calibration_slope'],brier=result['brier'],mean_probability=sum(p)/len(p),observed_fraction=sum(y)/len(y)))
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    with (out/'calibration_intercept_slope.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    (out/'calibration_diagnostics.json').write_text(json.dumps(dict(estimation='Intercept: logit probability offset with coefficient fixed at 1; slope: separate logistic regression with a free intercept. Original probabilities remain unchanged.',results=rows),indent=2),encoding='utf-8')
    print('Original-workflow calibration diagnostics reproduced:',rows,flush=True)
    return rows
if __name__=='__main__':main()
