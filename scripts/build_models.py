"""Reproduce the supplied Group 2 formulas; export auditable static app assets.
Run: uv run python scripts/build_models.py. Never executes notebook cells.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT=Path(__file__).resolve().parents[1]
DAYS=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
FORMULAS={
'Baseline':'bikes_hired ~ temp',
'A':'bikes_hired ~ temp + C(day_of_week)',
'B':'bikes_hired ~ temp + humidity + C(day_of_week)',
'C':'bikes_hired ~ temp + humidity + tempmax + dew + C(day_of_week)',
'D':'bikes_hired ~ temp + humidity + tempmax + dew + precip + C(day_of_week) + windspeed + visibility',
'E':'bikes_hired ~ temp + precip + C(season_name) + C(day_of_week) + windspeed + visibility + solarenergy + temp*precip + post_price_change'}
CHANGES={
'Baseline':'Temperature only', 'A':'+ Day of week', 'B':'+ Humidity', 'C':'+ Maximum temperature and dew point',
'D':'+ Precipitation, wind speed and visibility',
'E':'Redesigned: removes humidity, maximum temperature and dew point; adds season, solar energy, temperature × precipitation and post-price-change indicator'}
FEATURES={
'Baseline':['temp'],'A':['temp','day_of_week'],'B':['temp','humidity','day_of_week'],
'C':['temp','humidity','tempmax','dew','day_of_week'],
'D':['temp','humidity','tempmax','dew','precip','windspeed','visibility','day_of_week'],
'E':['temp','precip','windspeed','visibility','solarenergy','temp_precip','post_price_change','day_of_week','season_name']}

def export(model):
 out={}
 for term,value in model.params.items():
  if term.startswith('C(day_of_week)[T.'):term='day_'+term.split('[T.')[1][:-1]
  elif term.startswith('C(season_name)[T.'):term='season_'+term.split('[T.')[1][:-1]
  elif term=='temp:precip':term='temp_precip'
  out[term]=float(value)
 if 'C(day_of_week)' in model.model.formula:out['day_Mon']=0.
 if 'C(season_name)' in model.model.formula:out['season_Autumn']=0.
 return out

def main():
 df=pd.read_csv(ROOT/'data/london_bikes.csv')
 df['date']=pd.to_datetime(df.date,utc=True).dt.tz_localize(None)
 df=df[df.date>='2014-01-01'].copy()
 excluded=df[df.bikes_hired<=1][['date','bikes_hired']].copy();excluded['date']=excluded.date.dt.strftime('%Y-%m-%d')
 df=df[df.bikes_hired>1].copy()
 df['day_of_week']=pd.Categorical(df.day_of_week,categories=DAYS,ordered=True)
 df['post_price_change']=(df.date>='2022-09-12').astype(int)
 numeric=['temp','humidity','tempmax','dew','precip','windspeed','visibility','solarenergy']
 assert not df[numeric+['bikes_hired','season_name','day_of_week']].isna().any().any(), 'Need an explicitly shared complete-case sample'
 train=df[df.date<'2025-01-01'];test=df[df.date>='2025-01-01']
 predictions=df[['date','bikes_hired']].copy()
 holdout=test[['date','bikes_hired']].copy()
 artifacts={}
 for name,formula in FORMULAS.items():
  model=smf.ols(formula,data=df).fit()
  test_model=smf.ols(formula,data=train).fit();pred=test_model.predict(test)
  err=test.bikes_hired-pred
  predictions[name]=model.predict(df)
  holdout[name]=pred
  artifacts[name]={'formula':formula,'change':CHANGES[name],'features':FEATURES[name], 'coefficients':export(model),
     'n':int(model.nobs),'parameters':len(model.params),'adj_r2':float(model.rsquared_adj),'rse':float(model.scale**.5),
     'fit_rmse':float(np.mean(model.resid**2)**.5),'aic':float(model.aic),
     'holdout_rmse':float(np.mean(err**2)**.5),'holdout_mae':float(np.mean(abs(err))),
     'holdout_r2':float(1-np.sum(err**2)/np.sum((test.bikes_hired-test.bikes_hired.mean())**2)),
     'holdout_n':len(test),'holdout_coefficients':export(test_model)}
  if name=='E':
   supplied=pd.read_csv(ROOT/'model_coefficients.csv').set_index('term').coefficient.to_dict()
   reproduced=export(model)
   assert supplied.keys()==reproduced.keys()
   max_diff=max(abs(supplied[k]-reproduced[k]) for k in supplied)
   assert max_diff<1e-6, f'Final CSV mismatch: {max_diff}'
   artifacts[name]['coefficients']=supplied
   X=model.model.exog
   artifacts[name]['design_vif']=[{'term':t,'vif':float(variance_inflation_factor(X,i))} for i,t in enumerate(model.model.exog_names) if t!='Intercept']
   raw=df[['temp','solarenergy','precip','windspeed','visibility']]
   artifacts[name]['notebook_vif']=[{'term':t,'vif':float(variance_inflation_factor(raw.values,i))} for i,t in enumerate(raw.columns)]
 for frame,path in [(predictions,'model_fitted.csv'),(holdout,'model_holdout.csv')]:
  frame['date']=frame.date.dt.strftime('%Y-%m-%d');frame.to_csv(ROOT/'data'/path,index=False,float_format='%.8f')
 bundle={'models':artifacts,'sample':{'start':str(df.date.min().date()),'end':str(df.date.max().date()),'n':len(df),'excluded':excluded.to_dict('records')},
 'holdout':{'train_end':'2024-12-31','test_start':'2025-01-01','test_end':'2025-12-31','train_n':len(train),'test_n':len(test),
 'caveat':'Retrospective temporal check: formulas were already selected using 2014–2025. This is not an untouched final test. Uses observed weather, not archived advance weather forecasts.'},
 'final_csv_max_abs_difference':max_diff,'source_sha256':hashlib.sha256((ROOT/'data/london_bikes.csv').read_bytes()).hexdigest(),
 'coefficient_sha256':hashlib.sha256((ROOT/'model_coefficients.csv').read_bytes()).hexdigest(),
 'training_ranges':{c:{'min':float(df[c].min()),'max':float(df[c].max())} for c in numeric}}
 (ROOT/'data/model_comparison.json').write_text(json.dumps(bundle,indent=2,allow_nan=False))
 print(pd.DataFrame({k:{m:v[m] for m in ['adj_r2','rse','holdout_rmse','holdout_mae']} for k,v in artifacts.items()}).T.to_string())
 print('Final CSV max difference:',max_diff,'Excluded:',excluded.to_dict('records'))

if __name__=='__main__':main()
