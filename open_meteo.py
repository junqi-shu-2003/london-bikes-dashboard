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
    return frame[['date','day_of_week',*FIELDS]]

def fetch_weather(period, dates):
    # Fixed central London reference point; identical for history and forecast.
    params = dict(latitude=51.5085, longitude=-0.1257, timezone='Europe/London',
                  wind_speed_unit='kmh', temperature_unit='celsius', precipitation_unit='mm',
                  start_date=dates[0].strftime('%Y-%m-%d'), end_date=dates[-1].strftime('%Y-%m-%d'))
    history = period == 'january'
    params['hourly' if history else 'daily'] = ','.join(HOURLY if history else DAILY)
    url = 'https://archive-api.open-meteo.com/v1/archive' if history else 'https://api.open-meteo.com/v1/forecast'
    response = None
    for attempt in range(2):
        try:
            response = requests.get(url, params=params, timeout=(5,10))
            response.raise_for_status()
            break
        except requests.RequestException:
            # Retry a timeout, connection failure, 429 or server error once.
            if attempt or (response is not None and 400 <= response.status_code < 500 and response.status_code != 429):
                raise
            time.sleep(.3)
    payload = response.json()
    block = payload['hourly' if history else 'daily']
    frame = pd.DataFrame({col:block[field] for field,col in (HOURLY if history else DAILY).items()})
    frame['date'] = pd.to_datetime(block['time'])
    if history:
        expected = pd.date_range(dates[0].tz_localize(LONDON), (dates[-1]+pd.Timedelta(days=1)).tz_localize(LONDON), freq='h', inclusive='left').tz_localize(None)
        if frame.date.tolist() != list(expected) or frame[FIELDS].isna().any().any():
            raise ValueError('Historical hourly weather is incomplete')
        frame['date'] = frame.date.dt.normalize()
        frame = frame.groupby('date',as_index=False).agg(temp=('temp','mean'),humidity=('humidity','mean'),precip=('precip','sum'),windspeed=('windspeed','mean'),cloudcover=('cloudcover','mean'))
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
        key = f'{period}_{dates[0]:%Y-%m-%d}_{dates[-1]:%Y-%m-%d}'
        ttl = 86400 if period=='january' else 3600
        base = dict(period=period,start=f'{dates[0]:%Y-%m-%d}',end=f'{dates[-1]:%Y-%m-%d}',location='Central London, United Kingdom',timezone='Europe/London')
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
                entry = dict(rows=frame.to_dict('records'),fetched_at=now.astimezone(UTC).isoformat())
                self._write(key,entry)
                self.failures.pop(key,None)
                return dict(base,**entry,status='fresh')
            except (requests.RequestException,ValueError,KeyError,TypeError,OverflowError):
                LOG.warning('Weather request failed for %s',key,exc_info=True)
                self.failures[key] = now
                return dict(base,**(cached or {'rows':[],'fetched_at':None}),status='stale' if cached else 'unavailable')

weather_service = WeatherService()
