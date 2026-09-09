import numpy as np
import pandas as pd
import pytest
from models import BUNDLE,MODELS,predict
from model_views import FITTED,HOLDOUT,diagnostics
from app import show_weather,app
from open_meteo import period_dates,WeatherService,hourly_frame,LONDON,UTC
from test_weather import NOW,sample


def test_csv_matches_reproduction_and_notebook_metrics():
    assert BUNDLE['sample']['n']==4381
    assert BUNDLE['final_csv_max_abs_difference']<1e-6
    expected=[.3624,.4135,.4552,.4996,.5753,.6782]
    assert np.allclose([m['adj_r2'] for m in MODELS.values()],expected,atol=.00005)
    assert len(HOLDOUT)==365 and HOLDOUT.date.min()==pd.Timestamp('2025-01-01')
    assert BUNDLE['holdout']['train_n']==4016


def test_final_coefficient_arithmetic_and_calendar():
    row={'date':'2026-01-01','temp':10.,'humidity':70.,'precip':2.,'windspeed':5.,'windspeed_max':9.,'cloudcover':20.,'visibility':12.,'solarenergy':3.}
    c=MODELS['E']['coefficients']
    expected=c['Intercept']+10*c['temp']+2*c['precip']+9*c['windspeed']+12*c['visibility']+3*c['solarenergy']+c['post_price_change']+20*c['temp_precip']+c['day_Thu']+c['season_Winter']
    value,missing=predict([row],'E','maximum')
    assert value[0]==pytest.approx(expected) and missing==[]
    mean,_=predict([row],'E','mean')
    assert mean[0]-value[0]==pytest.approx((5-9)*c['windspeed'])
    assert np.isnan(predict([row],'E',None)[0][0])
    del row['visibility']
    assert np.isnan(predict([row],'E','maximum')[0][0])
    assert np.isfinite(predict([row],'B',None)[0][0])


def test_runtime_matches_all_fitted_values():
    raw=pd.read_csv('data/london_bikes.csv')
    raw=raw[(raw.date>='2014-01-01') & (raw.bikes_hired>1)]
    for model in MODELS:
        values,missing=predict(raw.to_dict('records'),model,'mean')
        assert not missing
        assert np.allclose(values,FITTED[model],atol=1e-6)


def test_hourly_dst_aggregation_boundaries():
    for start,hours in [('2026-03-29',23),('2026-10-25',25)]:
        dates=pd.date_range(start,periods=1)
        expected=pd.date_range(dates[0].tz_localize(LONDON),(dates[0]+pd.Timedelta(days=1)).tz_localize(LONDON),freq='h',inclusive='left')
        payload={'hourly':{'time':[int(t.timestamp()) for t in expected],'visibility':[1000]*len(expected)}}
        frame=hourly_frame(payload,{'visibility':'vis'},dates)
        assert len(frame)==hours and frame.date.nunique()==1
        payload['hourly']['time']=payload['hourly']['time'][:-1]
        payload['hourly']['visibility']=payload['hourly']['visibility'][:-1]
        with pytest.raises(ValueError):hourly_frame(payload,{'visibility':'vis'},dates)


def test_prediction_page_and_comparison_callbacks(tmp_path):
    def full(p,d):return sample(p,d).assign(visibility=15.,solarenergy=10.,windspeed_max=15.,tempmax=20.,dew=7.)
    result=WeatherService(tmp_path,full).get('january',now=NOW)
    shown=show_weather(result,'january','E','maximum',['Baseline','B'])
    assert len(shown[3].data)==3
    assert shown[4] not in ('—','Inputs needed')
    assert len(shown[1].children[1].children)==7
    empty=diagnostics('holdout',['E'],'2014-01-01','2014-12-31','E')
    assert 'No observations' in empty[3]
    diag=diagnostics('holdout',['Baseline','E'],'2025-01-01','2025-12-31','E')
    assert len(diag[0].data)==3 and '365 observed days' in diag[3]
    client=app.server.test_client()
    key=next(k for k in app.callback_map if k.startswith('..weather-status.children'))
    outputs=[{'id':o.component_id,'property':o.component_property} for o in app.callback_map[key]['output']]
    inputs=[{'id':id,'property':prop,'value':v} for id,prop,v in [('weather-result','data',result),('prediction-period','value','january'),('prediction-model','value','E'),('wind-basis','value','maximum'),('prediction-compare','value',['Baseline','B'])]]
    response=client.post('/_dash-update-component',json={'output':key,'outputs':outputs,'inputs':inputs,'state':[],'changedPropIds':['weather-result.data']})
    assert response.status_code==200
    assert response.json['response']['prediction-total']['children']==shown[4]
