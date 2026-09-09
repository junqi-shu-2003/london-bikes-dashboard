import pandas as pd
from app import app, DATA, trend_data, update_overview, update_scatter, all_years

def test_source_and_default_metrics():
    assert len(DATA)==4383
    assert not DATA.date.duplicated().any()
    result=update_overview('2025-01-01','2025-12-31','daily')
    assert result[:3]==('24,840','45,823','365')
    assert len(result[4].data[0].x)==365
    assert len(result[5].data[0].x)==7

def test_selected_partial_period_averages():
    frame=pd.DataFrame({'date':pd.to_datetime(['2025-01-03','2025-01-04','2025-01-06']), 'bikes_hired':[10.,30.,90.]})
    result=trend_data(frame,'weekly')
    assert result.bikes_hired.tolist()==[20.,90.]
    assert result.days.tolist()==[2,1]
    assert trend_data(frame,'monthly').bikes_hired.iloc[0]==130/3

def test_single_day_and_invalid_ranges():
    result=update_overview('2025-01-01','2025-01-01','daily')
    assert result[2]=='1'
    assert sum(pd.notna(result[5].data[0].y))==1
    for start,end in [(None,None),('2025-05-01','2025-01-01'),('2030-01-01','2030-01-02')]:
        result=update_overview(start,end,'daily')
        assert result[:3]==('—','—','0')
        assert len(result[4].layout.annotations)==1

def test_weather_and_colour_filters():
    for weather in ['temp','humidity','precip','windspeed','cloudcover']:
        for colour in ['day_type','season']:
            fig,note=update_scatter('2025-02-01','2025-02-28',weather,colour)
            assert sum(len(t.x) for t in fig.data)==28
            assert '28 paired' in note
    assert all_years(1)==('2014-01-01','2025-12-31')

def test_http_layout_and_real_callback():
    client=app.server.test_client()
    for path in ['/','/_dash-layout','/_dash-dependencies','/assets/style.css']:
        assert client.get(path).status_code==200
    key=next(k for k in app.callback_map if k.startswith('..average.children'))
    outputs=[{'id':o.component_id,'property':o.component_property} for o in app.callback_map[key]['output']]
    resp=client.post('/_dash-update-component',json={'output':key,'outputs':outputs,'inputs':[
        {'id':'dates','property':'start_date','value':'2025-01-01'},
        {'id':'dates','property':'end_date','value':'2025-01-31'},
        {'id':'frequency','property':'value','value':'monthly'}], 'state':[], 'changedPropIds':['dates.end_date']})
    assert resp.status_code==200
    assert resp.json['response']['days']['children']=='31'
