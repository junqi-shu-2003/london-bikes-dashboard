"""London bikes: observed daily demand. Run with uv run python app.py."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate
from open_meteo import weather_service, LONDON, FIELDS
from datetime import datetime
from models import MODELS, ORDER, LABELS, PALETTE, predict
import model_views

WEATHER = {'temp': 'Temperature (°C)', 'humidity': 'Relative humidity (%)',
           'precip': 'Precipitation (mm)', 'windspeed': 'Wind speed (km/h)',
           'cloudcover': 'Cloud cover (%)'}
DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
COLOURS = {'Weekday': '#087f79', 'Weekend': '#bc6227', 'Winter': '#426ca6',
           'Spring': '#42826a', 'Summer': '#b87924', 'Autumn': '#965976'}

def load_data(path):
    df = pd.read_csv(path)
    # These timestamps label calendar dates, not bike-event timestamps.
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='raise').dt.tz_localize(None).dt.normalize()
    df = df[df.date >= '2014-01-01'].copy()
    if df.date.duplicated().any():
        raise ValueError('Duplicate calendar dates in bike data.')
    for col in ['bikes_hired', *WEATHER]:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    df.loc[df.bikes_hired < 0, 'bikes_hired'] = np.nan
    df['day_of_week'] = df.date.dt.dayofweek.map(dict(enumerate(DAYS)))
    df['day_type'] = np.where(df.date.dt.dayofweek >= 5, 'Weekend', 'Weekday')
    df['season'] = df.date.dt.month.map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'})
    return df.sort_values('date').reset_index(drop=True)

DATA = load_data(Path(__file__).parent / 'data/london_bikes.csv')
MIN_DATE, MAX_DATE = DATA.date.min().date(), DATA.date.max().date()

def select_data(start, end):
    if not start or not end:
        return DATA.iloc[:0], 'Select both a start and an end date.'
    try:
        a, b = pd.Timestamp(start), pd.Timestamp(end)
        if a > b:
            return DATA.iloc[:0], 'Start date must be on or before end date.'
        frame = DATA[DATA.date.between(a, b)]
    except (ValueError, TypeError):
        return DATA.iloc[:0], 'Select a valid date range.'
    return frame, '' if len(frame) else 'No observations in this date range.'

def style(fig, ytitle='Daily hires'):
    fig.update_layout(template='plotly_white', font=dict(family='Arial, sans-serif',color='#243d4b',size=12),
        margin=dict(l=55,r=20,t=20,b=50), paper_bgcolor='white', plot_bgcolor='white',
        legend=dict(title=None,orientation='h',y=1.15,x=0), hoverlabel=dict(font_size=13))
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(title=ytitle,gridcolor='#edf1f3',rangemode='tozero',tickformat=',.0f')
    return fig

def empty(message):
    fig=style(go.Figure())
    fig.add_annotation(text=message,x=.5,y=.5,xref='paper',yref='paper',showarrow=False)
    fig.update_xaxes(visible=False);fig.update_yaxes(visible=False)
    return fig

def trend_data(frame, frequency):
    valid=frame.dropna(subset=['bikes_hired'])
    if frequency == 'daily':
        return valid[['date','bikes_hired']].assign(days=1)
    work=valid.assign(period=valid.date.dt.to_period('W-SUN' if frequency=='weekly' else 'M').dt.start_time)
    return work.groupby('period',as_index=False).agg(bikes_hired=('bikes_hired','mean'),days=('bikes_hired','count')).rename(columns={'period':'date'})

app=Dash(__name__,title='London Cycle Demand',update_title='Updating…')
server=app.server
CONFIG={'displaylogo':False,'scrollZoom':False,'modeBarButtonsToRemove':['lasso2d','select2d']}

def field(label, control):
    return html.Div([html.Label(label),control],className='field')

def stat(label, ident, caption):
    return html.Div([html.Div(label,className='muted'),html.Div(id=ident,className='value'),html.Div(caption,className='caption')],className='stat')

app.layout=html.Div([
    html.Header([html.Div([html.Div('London Cycle Demand',className='brand'),html.Div('Daily hire patterns & weather-based planning',className='caption')]),html.Span('Historical data · 2014–2025',className='badge')]),
    dcc.RadioItems(id='page',options=[{'label':'Explore','value':'explore'},{'label':'Predict','value':'predict'},{'label':'Models','value':'models'}],value='explore',className='navigation page-selector',inline=True),
    html.Main([
        html.H1('Understand daily demand'),html.P('London-wide hires · Historical data through 31 Dec 2025',className='muted'),
        html.Div([field('Date range',dcc.DatePickerRange(id='dates',start_date='2025-01-01',end_date='2025-12-31',min_date_allowed=MIN_DATE,max_date_allowed=MAX_DATE,display_format='D MMM YYYY',minimum_nights=0,first_day_of_week=1)),html.Button('All available years',id='all-years',n_clicks=0)],className='filters'),
        html.Div(id='range-note',className='caption',role='status',**{'aria-live':'polite'}),
        html.Div([stat('Average daily hires','average','Hires per valid day'),stat('Peak daily hires','peak','Highest daily count in selection'),stat('Valid days','days','Days with an observed hire count')],className='stats'),
        html.Section([html.Div([html.H2('Demand over time'),field('View',dcc.Dropdown(id='frequency',options=[{'label':'Daily','value':'daily'},{'label':'Weekly average','value':'weekly'},{'label':'Monthly average','value':'monthly'}],value='daily',clearable=False))],className='panel-head'),dcc.Graph(id='trend',config=CONFIG),html.Div(id='trend-note',className='caption')],className='panel'),
        html.Div([
            html.Section([html.H2('Weather & hires'),html.Div([field('Weather variable',dcc.Dropdown(id='weather',options=[{'label':v,'value':k} for k,v in WEATHER.items()],value='temp',clearable=False)),field('Colour by',dcc.Dropdown(id='colour',options=[{'label':'Weekday / weekend','value':'day_type'},{'label':'Season','value':'season'}],value='day_type',clearable=False))],className='weather-controls'),dcc.Graph(id='scatter',config=CONFIG),html.Div(id='scatter-note',className='caption')],className='panel'),
            html.Section([html.H2('Weekly pattern'),html.P('Average daily hires · Selected date range',className='caption'),dcc.Graph(id='weekday',config=CONFIG),html.Div('Hover for the number of observed days. Missing weekdays are not treated as zero.',className='caption')],className='panel')
        ],className='chart-grid'),
        html.Details([html.Summary('Data & definitions'),html.P('Source: course london_bikes.csv, copied locally without changes. Each row represents one London calendar day. Available exploration period: 1 Jan 2014–31 Dec 2025.'),html.P('Hires count rental transactions, not unique riders. Weekend means Saturday or Sunday; public holidays are not separately identified. Seasons: Dec–Feb winter, Mar–May spring, Jun–Aug summer, Sep–Nov autumn.'),html.P('KPIs and weekday means use valid hire counts. Scatter plots additionally require the selected weather value. Weekly and monthly views average observed days within the selected range, including partial periods. Empty groups are not filled with zero.'),html.P('Weather relationships show associations, not causal effects. Filters do not fit or retrain a model. Weather units follow the course helper convention; source measurement comparability will be checked before model integration.')]),
    ],id='explore-page'),
    html.Main([
        html.Div([html.Div([html.H1('Plan for expected demand'),html.P('London-wide daily hires · Weather-based planning',className='muted')]),html.Span('Group 2 · Model E connected',className='badge')],className='panel-head'),
        html.Div([dcc.RadioItems(id='prediction-period',options=[{'label':'Next five days','value':'future'},{'label':'1–7 Jan 2026','value':'january'}],value='future',inline=True,className='period-selector'),html.Button('Refresh weather',id='refresh-weather',n_clicks=0)],className='filters'),
        dcc.Interval(id='weather-clock',interval=60000,n_intervals=0),
        dcc.Store(id='weather-result'),dcc.Download(id='weather-download'),
        dcc.Loading(html.Div(id='weather-status',role='status',**{'aria-live':'polite'}),type='dot',color='#087f79'),
        html.Div([
            field('Prediction model',dcc.Dropdown(id='prediction-model',options=[{'label':LABELS[k],'value':k} for k in ORDER],value='E',clearable=False)),
            field('Training wind definition',dcc.Dropdown(id='wind-basis',options=[{'label':'Daily maximum · Visual Crossing convention','value':'maximum'},{'label':'Daily mean · Original course helper','value':'mean'}],placeholder='Choose the training wind definition…',clearable=True)),
            field('Compare model predictions',dcc.Dropdown(id='prediction-compare',options=[{'label':LABELS[k],'value':k} for k in ORDER],value=['Baseline','B'],multi=True))
        ],className='weather-controls'),
        html.Div([stat('Total predicted hires','prediction-total','Across the selected complete period'),stat('Average predicted hires','prediction-average','Hires per day'),stat('Busiest predicted day','prediction-busy','Highest daily estimate')],className='stats'),
        html.Section([html.H2('Daily hire estimates'),dcc.Graph(id='prediction-chart',config=CONFIG),html.Div(id='prediction-note',className='caption')],className='panel'),
        html.Section([html.Div([html.H2('Weather & prediction details'),html.Button('Download predictions CSV',id='download-weather',disabled=True,n_clicks=0)],className='panel-head'),html.Div(id='weather-table',className='weather-table'),html.P('Temperature, humidity, visibility and cloud are daily means. Solar energy and precipitation are daily totals. Wind uses your selected definition; the table shows daily mean until one is selected.',className='caption')],className='panel prediction-details'),
        html.Details([html.Summary('Weather source & model status'),html.P(['Weather: ',html.A('Open-Meteo',href='https://open-meteo.com/',target='_blank',rel='noopener noreferrer'),'. Fixed central London reference point: 51.5085° N, 0.1257° W. Calendar days use Europe/London, including daylight saving time.']),html.P('Next five days means tomorrow through the fifth day, excluding today. January uses historical reanalysis, supplemented with archived forecast visibility. This mixed weather source is labelled and is not a genuine advance forecast backtest. Actual January hire counts are unavailable in the course dataset.'),html.P('January weather is bundled with the app and needs no live request. Forecast weather is shared and cached for three hours; manual refresh has a 15-minute cooldown. Failed refreshes can show an older result only for the exact same dates, with its original timestamp. Requests pause after errors, including daily rate limits. Free hosting restarts may discard forecast cache and cooldown state; January remains available.'),html.P('Final E uses temperature, precipitation, wind, visibility (km), solar energy (MJ/m²), season, weekday, temperature × precipitation and a post-12-Sep-2022 indicator. Autumn and Monday are reference categories. The price-period indicator is 1 for both 2026 windows.'),html.P('Wind definition needs confirmation: Visual Crossing daily windspeed is a daily maximum; the original course helper used a mean. Neither convention is silently assumed. E and D predictions require a choice. Changing the selector changes weather inputs, not model coefficients.'),html.P('Models use the supplied full-sample coefficients. Missing required fields stop the affected predictions. Negative linear predictions are retained and flagged, not silently clipped. Point estimates do not include uncertainty from weather forecasts or coefficient estimation.')]),
    ],id='predict-page',style={'display':'none'}),
    model_views.layout(),
    html.Footer(['Course project · London cycle demand',html.Span('Calendar: Europe/London')])
],className='app-shell')

model_views.register(app)

@app.callback(Output('explore-page','style'),Output('predict-page','style'),Output('models-page','style'),Input('page','value'))
def change_page(page):
    return tuple({} if page==key else {'display':'none'} for key in ['explore','predict','models'])

@app.callback(Output('weather-result','data'),Input('page','value'),Input('prediction-period','value'),Input('refresh-weather','n_clicks'),Input('weather-clock','n_intervals'),running=[(Output('refresh-weather','disabled'),True,False)])
def retrieve_weather(page,period,clicks,ticks):
    if page!='predict':
        raise PreventUpdate
    return weather_service.get(period,force=ctx.triggered_id=='refresh-weather')

@app.callback(Output('weather-status','children'),Output('weather-table','children'),Output('download-weather','disabled'),Output('prediction-chart','figure'),Output('prediction-total','children'),Output('prediction-average','children'),Output('prediction-busy','children'),Output('prediction-note','children'),Input('weather-result','data'),Input('prediction-period','value'),Input('prediction-model','value'),Input('wind-basis','value'),Input('prediction-compare','value'))
def show_weather(result,selected_period,model='E',wind_basis=None,compare=None):
    blank=(empty('Awaiting complete model inputs'),'—','—','—','')
    if not result or result.get('period')!=selected_period:
        return ('Loading weather…',html.P('Waiting for weather data.',className='muted'),True,*blank)
    start,end=pd.Timestamp(result['start']),pd.Timestamp(result['end'])
    status=result['status']
    label={'snapshot':'Saved January weather · No live API required','fresh':'Weather updated','cached':'Cached weather','stale':'Weather update failed — showing an older cached result','unavailable':'Weather unavailable'}[status]
    timestamp='No weather retrieved.' if not result['fetched_at'] else 'Weather fetched '+datetime.fromisoformat(result['fetched_at']).astimezone(LONDON).strftime('%d %b %Y, %H:%M %Z')
    meaning='Reanalysis + archived forecast visibility · Not a forecast backtest' if selected_period=='january' else 'Tomorrow onward · Five complete London calendar days'
    extra=result.get('notice','')
    if result.get('retry_at'):
        extra += ' Next retry no earlier than '+datetime.fromisoformat(result['retry_at']).astimezone(LONDON).strftime('%d %b %Y, %H:%M %Z')+'. This is a retry time, not a guaranteed recovery time.'
    message=html.Div([html.Div(extra,className='caption'),html.Div(f'{start:%d %b %Y} – {end:%d %b %Y} · Central London'),html.Div(meaning,className='caption'),html.Div(label,className='weather-warning' if status in ('stale','unavailable') else 'weather-ok'),html.Div(timestamp,className='caption')])
    rows=result['rows']
    if not rows:
        return (message,html.P('No complete weather data is available for these dates.'),True,*blank)
    if model not in MODELS:model='E'
    values,missing=predict(rows,model,wind_basis)
    columns=[('temp','Temp °C'),('humidity','Humidity %'),('precip','Precip mm'),('windspeed_max' if wind_basis=='maximum' else 'windspeed','Wind km/h'),('cloudcover','Cloud %'),('visibility','Visibility km'),('solarenergy','Solar MJ/m²')]
    def number(value):
        return '—' if value is None or not np.isfinite(float(value)) else f'{value:.1f}'
    table=html.Table([html.Thead(html.Tr([html.Th(x,scope='col') for x in ['Date','Day',*[label for _,label in columns],f'Predicted hires · {model}']])),html.Tbody([html.Tr([html.Td(f"{pd.Timestamp(row['date']):%d %b %Y}"),html.Td(row['day_of_week']),*[html.Td(number(row.get(col))) for col,_ in columns],html.Td(f'{values[i]:,.0f}' if np.isfinite(values[i]) else 'Inputs needed')]) for i,row in enumerate(rows)])])
    fig=go.Figure();notes=[]
    selected=list(dict.fromkeys([model,*[k for k in (compare or []) if k in MODELS]]))
    for key in selected:
        pred,gaps=predict(rows,key,wind_basis)
        if not np.isfinite(pred).any():
            notes.append(f'{key} unavailable: '+(', '.join(gaps)))
            continue
        if key==model:fig.add_trace(go.Bar(x=[r['date'] for r in rows],y=pred,name=LABELS[key],marker_color=PALETTE[key],hovertemplate='%{x}<br>%{y:,.0f} hires<extra>%{fullData.name}</extra>'))
        else:fig.add_trace(go.Scatter(x=[r['date'] for r in rows],y=pred,name=LABELS[key],mode='lines+markers',line=dict(color=PALETTE[key],dash='dash'),hovertemplate='%{x}<br>%{y:,.0f} hires<extra>%{fullData.name}</extra>'))
    style(fig,'Predicted daily hires');fig.update_layout(hovermode='x unified');fig.update_xaxes(type='date',tickformat='%d %b',dtick=86400000)
    if not fig.data:fig=empty('Required model inputs are unavailable')
    if np.isfinite(values).all():
        total=f'{sum(values):,.0f}';average=f'{np.mean(values):,.0f}';busy=f"{pd.Timestamp(rows[int(np.argmax(values))]['date']):%d %b}"
    else:total=average=busy='Inputs needed'
    if missing and 'windspeed' in missing and wind_basis is None:notes.insert(0,'Confirm the training wind definition to enable D/E estimates.')
    if np.any(values<0):notes.append('Negative linear estimates are shown unaltered; they are not physically feasible hire counts.')
    for col in MODELS[model]['features']:
        bounds=model_views.BUNDLE['training_ranges'].get(col)
        if bounds:
            source='windspeed_max' if col=='windspeed' and wind_basis=='maximum' else col
            numeric=[r.get(source) for r in rows if r.get(source) is not None]
            if any(v<bounds['min'] or v>bounds['max'] for v in numeric):notes.append(f'{col} is outside its training range; extrapolation risk.')
    notes.append('Model '+model+' · Full-sample coefficients · Point estimates, not actual hires. Weather-source differences can affect accuracy.')
    return message,table,False,fig,total,average,busy,' '.join(notes)

@app.callback(Output('weather-download','data'),Input('download-weather','n_clicks'),State('weather-result','data'),State('prediction-period','value'),State('prediction-model','value'),State('wind-basis','value'),prevent_initial_call=True)
def download_weather(clicks,result,period,model='E',wind_basis=None):
    if not result or not result.get('rows') or result.get('period')!=period:
        raise PreventUpdate
    frame=pd.DataFrame(result['rows'])
    for key in ORDER:frame['predicted_hires_'+key]=predict(result['rows'],key,wind_basis)[0]
    frame['weather_fetched_at']=result['fetched_at'];frame['weather_status']=result['status']
    frame['location']=result['location'];frame['timezone']=result['timezone']
    frame['weather_source']=result.get('weather_source','Open-Meteo')
    frame['selected_model']=model;frame['wind_definition']=wind_basis or 'Unconfirmed'
    frame['model_status']='Coefficients connected; blank predictions indicate missing inputs'
    return dcc.send_data_frame(frame.to_csv,f"london_predictions_{result['start']}_{result['end']}.csv",index=False)

@app.callback(Output('dates','start_date'),Output('dates','end_date'),Input('all-years','n_clicks'),prevent_initial_call=True)
def all_years(_):
    return MIN_DATE.isoformat(),MAX_DATE.isoformat()

@app.callback(Output('average','children'),Output('peak','children'),Output('days','children'),Output('range-note','children'),Output('trend','figure'),Output('weekday','figure'),Output('trend-note','children'),Input('dates','start_date'),Input('dates','end_date'),Input('frequency','value'))
def update_overview(start,end,frequency):
    frame,error=select_data(start,end)
    valid=frame.dropna(subset=['bikes_hired'])
    if error or valid.empty:
        message=error or 'No valid hire counts in this range.'
        return '—','—','0',message,empty(message),empty(message),''
    series=trend_data(frame,frequency)
    fig=px.line(series,x='date',y='bikes_hired',custom_data=['days'])
    fig.update_traces(line_color='#087f79',line_width=2,mode='lines+markers' if len(series)<3 else 'lines',hovertemplate='%{x|%d %b %Y}<br>Hires: %{y:,.1f}<br>Observed days: %{customdata[0]}<extra></extra>')
    title='Daily hires' if frequency=='daily' else 'Average daily hires'
    style(fig,title)
    summary=valid.groupby('day_of_week').agg(hires=('bikes_hired','mean'),days=('bikes_hired','count')).reindex(DAYS)
    bar=go.Figure(go.Bar(x=DAYS,y=summary.hires,customdata=summary.days,marker_color='#087f79',hovertemplate='%{x}<br>Average hires: %{y:,.1f}<br>Observed days: %{customdata}<extra></extra>'))
    style(bar,'Average daily hires')
    note=f"{pd.Timestamp(start):%d %b %Y} – {pd.Timestamp(end):%d %b %Y} · {len(valid):,} valid days · {len(frame)-len(valid):,} missing hire counts"
    period_note='Each point is one observed day.' if frequency=='daily' else 'Averages use selected observed days; partial periods are included. Dates label the start of each period.'
    return f'{valid.bikes_hired.mean():,.0f}',f'{valid.bikes_hired.max():,.0f}',f'{len(valid):,}',note,fig,bar,period_note

@app.callback(Output('scatter','figure'),Output('scatter-note','children'),Input('dates','start_date'),Input('dates','end_date'),Input('weather','value'),Input('colour','value'))
def update_scatter(start,end,weather,colour):
    frame,error=select_data(start,end)
    if error:return empty(error),''
    if weather not in WEATHER or colour not in ('day_type','season'):return empty('Choose a weather variable and colour grouping.'),''
    valid=frame.dropna(subset=['bikes_hired',weather])
    if valid.empty:return empty('No paired weather and hire observations.'),'0 paired observations'
    fig=px.scatter(valid,x=weather,y='bikes_hired',color=colour,symbol=colour,
        category_orders={'day_type':['Weekday','Weekend'],'season':['Winter','Spring','Summer','Autumn']},color_discrete_map=COLOURS,
        hover_data={'date':'|%d %b %Y','day_of_week':True},labels={weather:WEATHER[weather],'bikes_hired':'Daily hires','date':'Date','day_of_week':'Day','day_type':'Day type','season':'Season'},opacity=.65)
    fig.update_traces(marker_size=7)
    style(fig);fig.update_xaxes(title=WEATHER[weather])
    return fig,f'{len(valid):,} paired observations · {len(frame)-len(valid):,} excluded for missing values · Association, not causation'

if __name__=='__main__':
    app.run(host='127.0.0.1',port=int(os.environ.get('PORT','8050')),debug=False)
