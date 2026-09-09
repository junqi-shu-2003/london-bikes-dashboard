from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html,dcc,Input,Output
from models import BUNDLE,MODELS,ORDER,LABELS,PALETTE
ROOT=Path(__file__).parent
FITTED=pd.read_csv(ROOT/'data/model_fitted.csv',parse_dates=['date'])
HOLDOUT=pd.read_csv(ROOT/'data/model_holdout.csv',parse_dates=['date'])

def style(fig):
    fig.update_layout(template='plotly_white',font=dict(family='Arial',color='#243d4b'),margin=dict(l=60,r=20,t=30,b=60),legend=dict(orientation='h',y=1.18),hovermode='closest')
    fig.update_xaxes(showgrid=False);fig.update_yaxes(gridcolor='#edf1f3')
    return fig

def layout():
    rows=[]
    for k,m in MODELS.items():
        rows.append(html.Tr([html.Th(LABELS[k],scope='row'),html.Td(m['change']),html.Td(str(m['parameters'])),html.Td(f"{m['adj_r2']:.3f}"),html.Td(f"{m['rse']:,.0f}"),html.Td(f"{m['holdout_rmse']:,.0f}"),html.Td(f"{m['holdout_mae']:,.0f}")]))
    return html.Main([
        html.H1('What changes when predictors change?'),html.P('Six Group 2 specifications · Same 4,381 observed days for full-sample fits',className='muted'),
        html.Div([html.Div([html.Div('Final model E · Adjusted R²',className='muted'),html.Div(f"{MODELS['E']['adj_r2']:.3f}",className='value'),html.Div('Full-sample fit · Higher is better',className='caption')],className='stat'),
        html.Div([html.Div('Final model E · Residual SE',className='muted'),html.Div(f"{MODELS['E']['rse']:,.0f}",className='value'),html.Div('Hires / day · Full-sample fit',className='caption')],className='stat'),
        html.Div([html.Div('Final model E · 2025 RMSE',className='muted'),html.Div(f"{MODELS['E']['holdout_rmse']:,.0f}",className='value'),html.Div('Retrospective temporal check · Lower is better',className='caption')],className='stat')],className='stats'),
        html.Div([html.Section([html.H2('Fit improves across the specifications'),dcc.Graph(id='model-fit',figure=fit_chart())],className='panel'),html.Section([html.H2('More variables do not always reduce test error'),dcc.Graph(id='model-error',figure=error_chart())],className='panel')],className='chart-grid'),
        html.P('2025 check: each formula is re-fitted on 2014–2024 and evaluated on 2025 observed weather. Formulas were already chosen using the full dataset, so this is a retrospective check, not an untouched final test or an evaluation of advance weather forecasts.',className='weather-warning'),
        html.Section([html.H2('Model progression'),html.Div(html.Table([html.Thead(html.Tr([html.Th(x) for x in ['Model','Change from prior specification','Parameters','Adj. R²','Residual SE','2025 RMSE','2025 MAE']])),html.Tbody(rows)]),className='weather-table'),html.P('E removes humidity, maximum temperature and dew point from D, then adds seasonal, solar, interaction and price-period terms. D and E are not nested.',className='caption')],className='panel'),
        html.Section([html.H2('Compare observed and modelled hires'),
        html.Div([html.Div([html.Label('Evaluation basis'),dcc.Dropdown(id='comparison-basis',options=[{'label':'2025 retrospective temporal check','value':'holdout'},{'label':'Full-sample fitted values','value':'fit'}],value='holdout',clearable=False)],className='field'),
        html.Div([html.Label('Models to compare'),dcc.Dropdown(id='comparison-models',options=[{'label':LABELS[k],'value':k} for k in ORDER],value=['Baseline','B','E'],multi=True)],className='field')],className='weather-controls'),
        html.Div([html.Label('Display range'),dcc.DatePickerRange(id='comparison-dates',start_date='2025-01-01',end_date='2025-12-31',min_date_allowed='2014-01-01',max_date_allowed='2025-12-31',minimum_nights=0,display_format='D MMM YYYY')],className='filters'),
        html.Div(id='comparison-note',className='caption'),dcc.Graph(id='comparison-series'),html.P('Monthly averages over the selected observed days. Date filters affect these diagnostic charts only; the six-model summary above stays fixed.',className='caption'),
        html.Div([html.Label('Inspect one model'),dcc.Dropdown(id='diagnostic-model',options=[{'label':LABELS[k],'value':k} for k in ORDER],value='E',clearable=False)],className='field'),
        html.Div([dcc.Graph(id='comparison-scatter'),dcc.Graph(id='comparison-residual')],className='chart-grid'),html.Div(id='model-formula',className='formula')],className='panel prediction-details'),
        html.Details([html.Summary('VIF audit & interpretation'),html.P('The supplied Notebook reports VIF without an intercept on five numeric columns. Its text says all values are below 5, but temperature, wind speed and visibility exceed 5. This should be corrected in the written submission.'),
        html.Div(html.Table([html.Thead(html.Tr([html.Th('Numeric predictor'),html.Th('Notebook VIF')])),html.Tbody([html.Tr([html.Td(r['term']),html.Td(f"{r['vif']:.2f}")]) for r in MODELS['E']['notebook_vif']])]),className='weather-table'),
        html.P('Additional audit: VIF below uses the final model’s complete design matrix, including its intercept, calendar indicators, interaction and price-period term. Intercept VIF is omitted. This differs from the Notebook calculation.'),
        html.Div(html.Table([html.Thead(html.Tr([html.Th('Design term'),html.Th('VIF')])),html.Tbody([html.Tr([html.Td(r['term']),html.Td(f"{r['vif']:.2f}")]) for r in MODELS['E']['design_vif']])]),className='weather-table'),
        html.P('With an interaction, temperature’s marginal association is 656.70 − 21.41 × precipitation hires per °C. The 656.70 main coefficient applies at zero precipitation. The post-price-change indicator is a date-period association, not a causal estimate of a price change.')]),
        html.Details([html.Summary('Reproduction & data scope'),html.P('Source: bikes_assignment_group2.ipynb and model_coefficients.csv. All six formulas were reproduced locally; final E coefficients match the supplied CSV within 1e-6. The app reads precomputed comparison results; page filters never retrain models.'),html.P('Modelling excludes 10–11 September 2022, each with zero hires, matching the Notebook. Explore retains these observed zero days, so its total is 4,383 rather than 4,381. The Notebook attributes the zeros to system closure; that explanation has not been independently verified.'),html.P('Residuals mean observed minus predicted hires. The original diagnostics note heavy tails and heteroscedasticity; point estimates should not be treated as precise operational allocations.')])
    ],id='models-page',style={'display':'none'})

def fit_chart():
    fig=go.Figure(go.Bar(x=ORDER,y=[MODELS[k]['adj_r2'] for k in ORDER],marker_color=[PALETTE[k] for k in ORDER],text=[f"{MODELS[k]['adj_r2']:.3f}" for k in ORDER],textposition='outside',hovertemplate='Model %{x}<br>Adjusted R²: %{y:.3f}<extra></extra>'))
    style(fig);fig.update_yaxes(title='Adjusted R²',range=[0,.8]);return fig

def error_chart():
    fig=go.Figure(go.Bar(x=ORDER,y=[MODELS[k]['holdout_rmse'] for k in ORDER],marker_color=[PALETTE[k] for k in ORDER],text=[f"{MODELS[k]['holdout_rmse']:,.0f}" for k in ORDER],textposition='outside',hovertemplate='Model %{x}<br>2025 RMSE: %{y:,.0f} hires<extra></extra>'))
    style(fig);fig.update_yaxes(title='2025 RMSE · hires/day',rangemode='tozero',range=[0,9000]);return fig

def diagnostics(basis,selected,start,end,model):
    frame=(HOLDOUT if basis=='holdout' else FITTED)
    selected=[k for k in (selected or []) if k in MODELS]
    if model not in MODELS:model='E'
    try:
        if not start or not end or pd.Timestamp(start)>pd.Timestamp(end):raise ValueError()
        frame=frame[frame.date.between(start,end)]
    except (ValueError,TypeError):frame=frame.iloc[:0]
    if frame.empty:
        fig=style(go.Figure());fig.add_annotation(text='No observations for this range and evaluation basis.',x=.5,y=.5,xref='paper',yref='paper',showarrow=False)
        return fig,fig,fig,'No observations. The temporal check covers 2025 only.',MODELS[model]['formula']
    monthly=frame.set_index('date')[['bikes_hired',*selected]].resample('MS').mean()
    series=go.Figure()
    for name in ['bikes_hired',*selected]:
        series.add_trace(go.Scatter(x=monthly.index,y=monthly[name],name='Observed' if name=='bikes_hired' else LABELS[name],mode='lines+markers',line=dict(color='#243d4b' if name=='bikes_hired' else PALETTE[name],width=3 if name in ('bikes_hired','E') else 2,dash='solid' if name=='bikes_hired' else 'dash'),hovertemplate='%{x|%b %Y}<br>%{y:,.0f} hires/day<extra>%{fullData.name}</extra>'))
    style(series);series.update_layout(hovermode='x unified');series.update_yaxes(title='Average daily hires');series.update_xaxes(title='Month')
    scatter=go.Figure(go.Scatter(x=frame[model],y=frame.bikes_hired,mode='markers',marker=dict(color=PALETTE[model],opacity=.5,size=6),customdata=frame.date.dt.strftime('%d %b %Y'),hovertemplate='%{customdata}<br>Predicted: %{x:,.0f}<br>Observed: %{y:,.0f}<extra></extra>'))
    lo=min(frame[model].min(),frame.bikes_hired.min());hi=max(frame[model].max(),frame.bikes_hired.max())
    scatter.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',line=dict(color='#84949e',dash='dash'),name='Perfect prediction',hoverinfo='skip'))
    style(scatter);scatter.update_layout(showlegend=False,title='Observed vs predicted');scatter.update_xaxes(title='Predicted hires');scatter.update_yaxes(title='Observed hires')
    residual=go.Figure(go.Scatter(x=frame[model],y=frame.bikes_hired-frame[model],mode='markers',marker=dict(color=PALETTE[model],opacity=.5,size=6),hovertemplate='Predicted: %{x:,.0f}<br>Residual: %{y:,.0f}<extra></extra>'))
    residual.add_hline(y=0,line_color='#84949e');style(residual);residual.update_layout(title='Residuals vs predicted');residual.update_xaxes(title='Predicted hires');residual.update_yaxes(title='Observed − predicted hires')
    errors=frame.bikes_hired-frame[model]
    note=f"{len(frame):,} observed days · {LABELS[model]} selected-range RMSE: {np.mean(errors**2)**.5:,.0f} · {'Retrospective temporal check' if basis=='holdout' else 'In-sample fit, not forecast validation'}"
    return series,scatter,residual,note,MODELS[model]['formula']

def register(app):
    app.callback(Output('comparison-series','figure'),Output('comparison-scatter','figure'),Output('comparison-residual','figure'),Output('comparison-note','children'),Output('model-formula','children'),Input('comparison-basis','value'),Input('comparison-models','value'),Input('comparison-dates','start_date'),Input('comparison-dates','end_date'),Input('diagnostic-model','value'))(diagnostics)
