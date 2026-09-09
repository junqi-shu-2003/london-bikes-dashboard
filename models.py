"""Strict coefficient inference. Training only occurs in scripts/build_models.py."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).parent
BUNDLE=json.loads((ROOT/'data/model_comparison.json').read_text())
MODELS=BUNDLE['models']
if hashlib.sha256((ROOT/'model_coefficients.csv').read_bytes()).hexdigest()!=BUNDLE['coefficient_sha256']:
    raise ValueError('Final coefficients changed: rebuild and verify comparison artifacts first')
MODELS['E']['coefficients']=pd.read_csv(ROOT/'model_coefficients.csv').set_index('term').coefficient.to_dict()
ORDER=list(MODELS)
LABELS={k:('Baseline · Temperature' if k=='Baseline' else f'Model {k}'+(' · Group 2 final' if k=='E' else '')) for k in ORDER}
PALETTE={'Baseline':'#84949e','A':'#5587ac','B':'#ac6b23','C':'#965d9b','D':'#b85855','E':'#087f79'}

def features(rows,wind_basis=None):
    frame=pd.DataFrame(rows).copy()
    if frame.empty:return frame
    frame['date']=pd.to_datetime(frame.date)
    day=frame.date.dt.dayofweek
    for i,name in enumerate(['Mon','Tue','Wed','Thu','Fri','Sat','Sun']):frame['day_'+name]=(day==i).astype(int)
    season=frame.date.dt.month.map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'})
    for name in ['Autumn','Spring','Summer','Winter']:frame['season_'+name]=(season==name).astype(int)
    frame['post_price_change']=(frame.date>='2022-09-12').astype(int)
    frame['temp_precip']=pd.to_numeric(frame['temp'],errors='coerce')*pd.to_numeric(frame['precip'],errors='coerce')
    frame['Intercept']=1.
    if wind_basis=='maximum':frame['windspeed']=pd.to_numeric(frame.get('windspeed_max',np.nan),errors='coerce')
    elif wind_basis!='mean':frame['windspeed']=np.nan
    return frame

def predict(rows,model='E',wind_basis=None):
    frame=features(rows,wind_basis)
    if frame.empty:return np.array([]),[]
    coeff=MODELS[model]['coefficients']
    result=np.zeros(len(frame));valid=np.ones(len(frame),dtype=bool);missing=[]
    for term,value in coeff.items():
        values=pd.to_numeric(frame[term],errors='coerce').to_numpy(dtype=float) if term in frame else np.full(len(frame),np.nan)
        finite=np.isfinite(values)
        if not finite.all():missing.append(term)
        valid &= finite
        result+=value*values
    result[~valid]=np.nan
    return result,missing
