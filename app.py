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
    dcc.RadioItems(id='page',options=[{'label':'Explore','value':'explore'},{'label':'Predict','value':'predict'}],value='explore',className='navigation page-selector',inline=True),
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
        html.Div([html.Div([html.H1('Plan for expected demand'),html.P('London-wide daily hires · Weather-based planning',className='muted')]),html.Span('Model pending',className='badge')],className='panel-head'),
        html.Div([dcc.RadioItems(id='prediction-period',options=[{'label':'Next five days','value':'future'},{'label':'1–7 Jan 2026','value':'january'}],value='future',inline=True,className='period-selector'),html.Button('Refresh weather',id='refresh-weather',n_clicks=0)],className='filters'),
        dcc.Interval(id='weather-clock',interval=60000,n_intervals=0),
        dcc.Store(id='weather-result'),dcc.Download(id='weather-download'),
        dcc.Loading(html.Div(id='weather-status',role='status',**{'aria-live':'polite'}),type='dot',color='#087f79'),
        html.Div([html.Div([html.Div(label,className='muted'),html.Div('Model pending',className='pending-value'),html.Div(caption,className='caption')],className='stat') for label,caption in [('Total predicted hires','Across the selected period'),('Average predicted hires','Hires per day'),('Busiest predicted day','Highest daily estimate')]],className='stats'),
        html.Section([html.H2('Daily hire estimates'),html.Div([html.H3('Waiting for model coefficients'),html.P('Weather is available below. Hire predictions will appear once the model is connected.')],className='model-empty')],className='panel'),
        html.Section([html.Div([html.H2('Weather & prediction details'),html.Button('Download weather CSV',id='download-weather',disabled=True,n_clicks=0)],className='panel-head'),html.Div(id='weather-table',className='weather-table'),html.P('Temperature, humidity, wind and cloud are daily means; precipitation is a daily total.',className='caption')],className='panel prediction-details'),
        html.Details([html.Summary('Weather source & model status'),html.P(['Weather: ',html.A('Open-Meteo',href='https://open-meteo.com/',target='_blank',rel='noopener noreferrer'),'. Fixed central London reference point: 51.5085° N, 0.1257° W. Calendar days use Europe/London, including daylight saving time.']),html.P('Next five days means tomorrow through the fifth day, excluding today. January 2026 uses historical reanalysis weather, not forecasts issued before that week. Actual January hire counts are not available in the course dataset.'),html.P('Forecast weather is cached for one hour; historical weather for 24 hours. Refresh requests new weather. If an update fails, a previous complete result for the same dates may be shown with its original fetch time and a warning. A failed request without a matching cache shows no weather values.'),html.P('Model pending: supported inputs are temperature, humidity, precipitation, wind speed, cloud cover and day of week. The final model can use a subset. No hire estimates or prediction intervals are fabricated.')]),
    ],id='predict-page',style={'display':'none'}),
    html.Footer(['Course project · London cycle demand',html.Span('Calendar: Europe/London')])
],className='app-shell')

@app.callback(Output('explore-page','style'),Output('predict-page','style'),Input('page','value'))
def change_page(page):
    return ({'display':'none'}, {}) if page=='predict' else ({}, {'display':'none'})

@app.callback(Output('weather-result','data'),Input('page','value'),Input('prediction-period','value'),Input('refresh-weather','n_clicks'),Input('weather-clock','n_intervals'),running=[(Output('refresh-weather','disabled'),True,False)])
def retrieve_weather(page,period,clicks,ticks):
    if page!='predict':
        raise PreventUpdate
    return weather_service.get(period,force=ctx.triggered_id=='refresh-weather')

@app.callback(Output('weather-status','children'),Output('weather-table','children'),Output('download-weather','disabled'),Input('weather-result','data'),Input('prediction-period','value'))
def show_weather(result,selected_period):
    if not result or result.get('period')!=selected_period:
        return 'Loading weather…',html.P('Waiting for weather data.',className='muted'),True
    start,end=pd.Timestamp(result['start']),pd.Timestamp(result['end'])
    status=result['status']
    label={'fresh':'Weather updated','cached':'Cached weather','stale':'Weather update failed — showing an older cached result','unavailable':'Weather unavailable — please retry'}[status]
    timestamp='No weather retrieved.' if not result['fetched_at'] else 'Weather fetched '+datetime.fromisoformat(result['fetched_at']).astimezone(LONDON).strftime('%d %b %Y, %H:%M %Z')
    meaning='Historical weather-based estimate period · Not a forecast backtest' if selected_period=='january' else 'Tomorrow onward · Five complete London calendar days'
    message=html.Div([html.Div(f'{start:%d %b %Y} – {end:%d %b %Y} · Central London'),html.Div(meaning,className='caption'),html.Div(label,className='weather-warning' if status in ('stale','unavailable') else 'weather-ok'),html.Div(timestamp,className='caption')])
    rows=result['rows']
    if not rows:
        return message,html.P('No complete weather data is available for these dates. Hire predictions remain pending.'),True
    headings=['Date','Day','Temp °C','Humidity %','Precip mm','Wind km/h','Cloud %','Predicted hires']
    table=html.Table([html.Thead(html.Tr([html.Th(x,scope='col') for x in headings])),html.Tbody([html.Tr([html.Td(f"{pd.Timestamp(row['date']):%d %b %Y}"),html.Td(row['day_of_week']),*[html.Td(f"{row[col]:.1f}") for col in FIELDS],html.Td('Model pending',className='muted')]) for row in rows])])
    return message,table,False

@app.callback(Output('weather-download','data'),Input('download-weather','n_clicks'),State('weather-result','data'),State('prediction-period','value'),prevent_initial_call=True)
def download_weather(clicks,result,period):
    if not result or not result.get('rows') or result.get('period')!=period:
        raise PreventUpdate
    frame=pd.DataFrame(result['rows'])
    frame['weather_fetched_at']=result['fetched_at']
    frame['weather_status']=result['status']
    frame['location']=result['location']
    frame['timezone']=result['timezone']
    frame['model_status']='Model pending'
    return dcc.send_data_frame(frame.to_csv,f"london_weather_{result['start']}_{result['end']}.csv",index=False)

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
