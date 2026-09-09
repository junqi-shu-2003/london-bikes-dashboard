"""Validated London weather, with bounded retries and a persistent cache.

Cache keys include exact calendar dates: yesterday's five-day window is never
served as today's. No calls are made at import time.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from threading import RLock
import json
import logging
import os
import tempfile
import time

import numpy as np
import pandas as pd
import requests

LONDON = ZoneInfo('Europe/London')
UTC = ZoneInfo('UTC')
FIELDS = ['temp', 'humidity', 'precip', 'windspeed', 'cloudcover']
DAILY = dict(zip(['temperature_2m_mean','relative_humidity_2m_mean','precipitation_sum','wind_speed_10m_mean','cloud_cover_mean'], FIELDS))
HOURLY = dict(zip(['temperature_2m','relative_humidity_2m','precipitation','wind_speed_10m','cloud_cover'], FIELDS))
LOG = logging.getLogger(__name__)

def period_dates(period, now=None):
    now = now or datetime.now(UTC)
    if period == 'january':
        return pd.date_range('2026-01-01', '2026-01-07')
    if period != 'future':
        raise ValueError('Unknown weather period')
    start = now.astimezone(LONDON).date() + timedelta(days=1)
    return pd.date_range(start, periods=5)

def validate(frame, dates):
    frame = frame.copy()
    frame['date'] = pd.to_datetime(frame['date']).dt.normalize()
    frame = frame.sort_values('date').reset_index(drop=True)
    if frame.date.tolist() != list(dates):
        raise ValueError('Weather dates are incomplete or duplicated')
    for col in FIELDS:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
    if not np.isfinite(frame[FIELDS].to_numpy(dtype=float)).all():
        raise ValueError('Weather contains missing or non-finite values')
    if not frame.humidity.between(0,100).all() or not frame.cloudcover.between(0,100).all():
        raise ValueError('Weather percentages outside 0–100')
    if (frame[['precip','windspeed']] < 0).any().any():
        raise ValueError('Negative precipitation or wind speed')
    frame['day_of_week'] = frame.date.dt.dayofweek.map(dict(enumerate(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])))
    extras = [c for c in ['tempmax','dew','solarenergy','visibility','windspeed_max'] if c in frame]
    for col in extras:
        frame[col] = pd.to_numeric(frame[col], errors='coerce').replace([np.inf,-np.inf], np.nan)
    return frame[['date','day_of_week',*FIELDS,*extras]]

def request_json(url, params):
    response = None
    for attempt in range(2):
        try:
            response = requests.get(url, params=params, timeout=(5, 12))
            if not response.ok:
                LOG.warning("Open-Meteo HTTP %s: %s", response.status_code, response.text[:400])
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt or (response is not None and 400 <= response.status_code < 500 and response.status_code != 429):
                raise
            time.sleep(.3)


def hourly_frame(payload, mapping, dates):
    block = payload['hourly']
    frame = pd.DataFrame({col:block[field] for field,col in mapping.items()})
    # Unix timestamps make London's 23/25-hour DST days unambiguous.
    frame['instant'] = pd.to_datetime(block['time'], unit='s', utc=True)
    expected = pd.date_range(dates[0].tz_localize(LONDON), (dates[-1]+pd.Timedelta(days=1)).tz_localize(LONDON), freq='h', inclusive='left').tz_convert(UTC)
    frame = frame[frame.instant.between(expected[0],expected[-1])].copy()
    if frame.instant.tolist() != list(expected):
        raise ValueError('Hourly weather coverage is incomplete')
    frame['date'] = frame.instant.dt.tz_convert(LONDON).dt.tz_localize(None).dt.normalize()
    return frame


def fetch_weather(period, dates):
    first=dates[0].tz_localize(LONDON).tz_convert(UTC)
    last=((dates[-1]+pd.Timedelta(days=1)).tz_localize(LONDON).tz_convert(UTC)-pd.Timedelta(hours=1))
    params = dict(latitude=51.5085, longitude=-0.1257, timezone='GMT', timeformat='unixtime',
                  wind_speed_unit='kmh', temperature_unit='celsius', precipitation_unit='mm',
                  start_date=first.strftime('%Y-%m-%d'), end_date=last.strftime('%Y-%m-%d'))
    history = period == 'january'
    mapping = dict(HOURLY, dew_point_2m='dew', shortwave_radiation='radiation')
    if not history:
        mapping['visibility'] = 'visibility_m'
    params['hourly'] = ','.join(mapping)
    url = 'https://archive-api.open-meteo.com/v1/archive' if history else 'https://api.open-meteo.com/v1/forecast'
    hours = hourly_frame(request_json(url, params), mapping, dates)
    if hours[FIELDS].isna().any().any():
        raise ValueError('Core hourly weather has missing values')
    frame = hours.groupby('date', as_index=False).agg(
        temp=('temp','mean'),tempmax=('temp','max'),humidity=('humidity','mean'),
        precip=('precip','sum'),windspeed=('windspeed','mean'),windspeed_max=('windspeed','max'),
        cloudcover=('cloudcover','mean'),dew=('dew',lambda s:s.mean(skipna=False)),
        solarenergy=('radiation',lambda s:s.sum(skipna=False)*.0036))
    if history:
        try:
            # Reanalysis does not supply visibility. Supplement only this field
            # with explicitly labelled archived forecast data; never fabricate it.
            extra_params = dict(params, hourly='visibility')
            vis = hourly_frame(request_json('https://historical-forecast-api.open-meteo.com/v1/forecast',extra_params),{'visibility':'visibility_m'},dates)
            visibility = vis.groupby('date').visibility_m.agg(lambda s:s.mean(skipna=False)/1000)
        except (requests.RequestException, ValueError, KeyError, TypeError):
            LOG.warning('Historical visibility supplement unavailable',exc_info=True)
            visibility = pd.Series(dtype=float)
    else:
        visibility = hours.groupby('date').visibility_m.agg(lambda s:s.mean(skipna=False)/1000)
    frame['visibility'] = frame.date.map(visibility)
    return validate(frame, dates)

class WeatherService:
    def __init__(self, cache_dir=None, fetcher=None):
        self.cache_dir = Path(cache_dir or Path(__file__).parent / '.weather_cache')
        self.fetcher = fetcher or fetch_weather
        self.memory = {}
        self.failures = {}
        self.lock = RLock()

    def _read(self, key, dates):
        try:
            entry = self.memory.get(key)
            if entry is None:
                entry = json.loads((self.cache_dir / (key+'.json')).read_text())
            validate(pd.DataFrame(entry['rows']),dates)
            datetime.fromisoformat(entry['fetched_at']).astimezone(UTC)
            return entry
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write(self, key, entry):
        self.memory[key] = entry
        path = None
        try:
            self.cache_dir.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w',dir=self.cache_dir,delete=False) as file:
                path = file.name
                json.dump(entry,file,allow_nan=False)
            os.replace(path,self.cache_dir/(key+'.json'))
        except OSError:
            LOG.warning('Disk cache unavailable; using memory cache',exc_info=True)
        finally:
            if path and os.path.exists(path):
                os.unlink(path)

    def get(self, period, force=False, now=None):
        now = now or datetime.now(UTC)
        dates = period_dates(period,now)
        key = f'v2_{period}_{dates[0]:%Y-%m-%d}_{dates[-1]:%Y-%m-%d}'
        ttl = 86400 if period=='january' else 3600
        base = dict(period=period,start=f'{dates[0]:%Y-%m-%d}',end=f'{dates[-1]:%Y-%m-%d}',location='Central London, United Kingdom',timezone='Europe/London', weather_source='Open-Meteo reanalysis + archived forecast visibility' if period=='january' else 'Open-Meteo live forecast')
        with self.lock:
            cached = self._read(key,dates)
            fresh = cached and 0 <= (now-datetime.fromisoformat(cached['fetched_at'])).total_seconds() < ttl
            recent_failure = key in self.failures and (now-self.failures[key]).total_seconds() < 60
            if recent_failure and not force:
                return dict(base,**(cached or {'rows':[],'fetched_at':None}),status='stale' if cached else 'unavailable')
            if fresh and not force:
                return dict(base,**cached,status='cached')
            try:
                frame = validate(self.fetcher(period,dates),dates)
                frame['date'] = frame.date.dt.strftime('%Y-%m-%d')
                rows = frame.astype(object).where(pd.notna(frame), None).to_dict('records')
                entry = dict(rows=rows,fetched_at=now.astimezone(UTC).isoformat())
                self._write(key,entry)
                self.failures.pop(key,None)
                return dict(base,**entry,status='fresh')
            except (requests.RequestException,ValueError,KeyError,TypeError,OverflowError):
                LOG.warning('Weather request failed for %s',key,exc_info=True)
                self.failures[key] = now
                return dict(base,**(cached or {'rows':[],'fetched_at':None}),status='stale' if cached else 'unavailable')

weather_service = WeatherService()
