from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import pytest
import requests
from open_meteo import WeatherService,period_dates,validate,FIELDS,fetch_weather
from app import show_weather,download_weather,change_page,app

NOW=datetime(2026,9,9,15,0,tzinfo=ZoneInfo('UTC'))

def sample(period,dates):
    return pd.DataFrame({'date':dates,'temp':12.,'humidity':65.,'precip':1.,'windspeed':10.,'cloudcover':40.})

def test_london_dates_and_dst():
    assert period_dates('future',NOW).strftime('%Y-%m-%d').tolist()==['2026-09-10','2026-09-11','2026-09-12','2026-09-13','2026-09-14']
    # 23:30 UTC is the next London calendar day during BST.
    assert str(period_dates('future',NOW.replace(hour=23,minute=30))[0].date())=='2026-09-11'
    assert len(period_dates('january',NOW))==7
    assert len(period_dates('future',datetime(2026,10,24,23,30,tzinfo=ZoneInfo('UTC'))))==5

def test_cache_persistence_expiry_and_force(tmp_path):
    calls=[]
    def fetch(p,d):calls.append(p);return sample(p,d)
    svc=WeatherService(tmp_path,fetch)
    first=svc.get('future',now=NOW)
    assert first['status']=='fresh'
    assert svc.get('future',now=NOW+timedelta(minutes=20))['status']=='cached'
    assert len(calls)==1
    assert WeatherService(tmp_path,fetch).get('future',now=NOW+timedelta(minutes=30))['status']=='cached'
    assert svc.get('future',now=NOW+timedelta(hours=3))['status']=='fresh'
    assert svc.get('future',force=True,now=NOW+timedelta(hours=3,minutes=16))['status']=='fresh'
    assert len(calls)==3
    svc.get('january',now=NOW)
    assert svc.get('january',now=NOW+timedelta(hours=2))['status']=='cached'
    assert svc.get('january',now=NOW+timedelta(hours=24))['status']=='fresh'

def test_failure_fallback_and_midnight(tmp_path):
    svc=WeatherService(tmp_path,sample)
    good=svc.get('future',now=NOW)
    def fail(*args):raise requests.Timeout('test outage')
    svc.fetcher=fail
    result=svc.get('future',force=True,now=NOW+timedelta(minutes=16))
    assert result['status']=='stale' and result['rows']==good['rows']
    assert result['fetched_at']==good['fetched_at']
    assert svc.get('future',now=NOW+timedelta(minutes=16,seconds=10))['status']=='stale'
    # Previous window must not be returned for a different London date.
    assert svc.get('future',now=NOW+timedelta(days=1))['status']=='unavailable'
    empty=WeatherService(tmp_path/'empty',fail).get('january',now=NOW)
    assert empty['rows']==[] and empty['fetched_at'] is None

def test_bad_data_rejected_without_overwriting_cache(tmp_path):
    svc=WeatherService(tmp_path,sample)
    first=svc.get('january',now=NOW)
    def bad(p,d):
        frame=sample(p,d);frame.loc[0,'humidity']=None;return frame
    svc.fetcher=bad
    assert svc.get('january',force=True,now=NOW)['rows']==first['rows']
    dates=period_dates('future',NOW)
    with pytest.raises(ValueError):validate(sample('future',dates).iloc[:-1],dates)
    with pytest.raises(ValueError):validate(pd.concat([sample('future',dates),sample('future',dates).iloc[:1]]),dates)

def test_display_download_and_tabs(tmp_path):
    result=WeatherService(tmp_path,sample).get('future',now=NOW)
    status,table,disabled,*_=show_weather(result,'future')
    assert not disabled
    assert len(table.children[1].children)==5
    assert show_weather(result,'january')[2]
    download=download_weather(1,result,'future')
    assert 'weather_fetched_at' in download['content']
    assert 'predicted_hires_E' in download['content']
    assert change_page('predict')==({'display':'none'}, {}, {'display':'none'})

def test_retry_and_exact_forecast_window(monkeypatch):
    calls=[]
    dates=period_dates('future',NOW)
    class Response:
        ok=True
        def raise_for_status(self):pass
        def json(self):
            times=pd.date_range('2026-09-09','2026-09-15',freq='h',inclusive='left',tz='UTC')
            return {'hourly':{'time':[int(x.timestamp()) for x in times],**{field:[value]*len(times) for field,value in {'temperature_2m':12,'relative_humidity_2m':60,'precipitation':1,'wind_speed_10m':10,'cloud_cover':50,'dew_point_2m':5,'shortwave_radiation':100,'visibility':10000}.items()}}}
    def get(url,params,timeout):
        calls.append(params)
        if len(calls)==1:raise requests.Timeout()
        return Response()
    monkeypatch.setattr('open_meteo.requests.get',get)
    assert len(fetch_weather('future',dates))==5
    assert len(calls)==2
    assert calls[-1]['start_date']=='2026-09-09' and calls[-1]['end_date']=='2026-09-14'
    assert calls[-1]['timezone']=='GMT'

def test_predict_http_callback(monkeypatch,tmp_path):
    monkeypatch.setattr('app.weather_service',WeatherService(tmp_path,sample))
    response=app.server.test_client().post('/_dash-update-component',json={
        'output':'weather-result.data','outputs':{'id':'weather-result','property':'data'},
        'inputs':[{'id':'page','property':'value','value':'predict'},{'id':'prediction-period','property':'value','value':'january'},{'id':'refresh-weather','property':'n_clicks','value':0},{'id':'weather-clock','property':'n_intervals','value':0}],
        'state':[],'changedPropIds':['page.value']})
    assert response.status_code==200
    assert len(response.json['response']['weather-result']['data']['rows'])==7

def test_bundled_january_offline(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No network allowed')
    monkeypatch.setattr('open_meteo.requests.get',forbidden)
    result=WeatherService(tmp_path).get('january',force=True,now=NOW)
    assert result['status']=='snapshot' and len(result['rows'])==7
    assert all(row['visibility'] is not None for row in result['rows'])
    assert not show_weather(result,'january','E','maximum')[2]

def test_daily_limit_persisted_and_not_bypassed(tmp_path):
    calls=[]
    def limited(*args):
        calls.append(1)
        r=requests.Response();r.status_code=429;r._content=b'{"reason":"Daily API request limit exceeded"}'
        raise requests.HTTPError(response=r)
    svc=WeatherService(tmp_path,limited)
    first=svc.get('future',now=NOW)
    assert first['status']=='unavailable' and 'rate limit' in first['notice']
    assert datetime.fromisoformat(first['retry_at'])==NOW+timedelta(days=1)
    svc.get('future',force=True,now=NOW+timedelta(hours=1))
    WeatherService(tmp_path,limited).get('future',force=True,now=NOW+timedelta(hours=2))
    assert len(calls)==1

def test_manual_refresh_cooldown(tmp_path):
    calls=[]
    def fetch(p,d):calls.append(1);return sample(p,d)
    svc=WeatherService(tmp_path,fetch)
    svc.get('future',now=NOW)
    assert svc.get('future',force=True,now=NOW+timedelta(minutes=14))['status']=='cached'
    assert len(calls)==1
    assert svc.get('future',force=True,now=NOW+timedelta(minutes=15))['status']=='fresh'
    assert len(calls)==2

def test_429_not_immediately_retried(monkeypatch):
    from open_meteo import request_json
    calls=[]
    def get(*args,**kwargs):
        calls.append(1)
        r=requests.Response();r.status_code=429;r._content=b'Daily API request limit exceeded'
        return r
    monkeypatch.setattr('open_meteo.requests.get',get)
    with pytest.raises(requests.HTTPError):request_json('https://example.test',{})
    assert len(calls)==1
