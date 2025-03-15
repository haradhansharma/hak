#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import math
import colorsys
import logging as d_logging
from datetime import datetime, timedelta
import shutil
import geopandas as gpd
import pandas as pd
import numpy as np


# [haradhan] Start

# Should run on non GUI mode
import matplotlib
matplotlib.use("Agg")
from django.core.cache import caches
from django.utils.crypto import get_random_string

# [haradhan] End

import matplotlib.pyplot as plt
import folium
from shapely.geometry import LineString
from folium.plugins import BoatMarker
import requests_cache
from retry_requests import retry
import openmeteo_requests


# =============================================================================
# Constants and Global Settings
# =============================================================================
TIMEZONE = 1
POWER_CALC_OPTION = False  # True = use interpolation, False = normal calculation

BATTERY_CAPACITY = 2752  # in kWh
HOTEL_LOAD_POWER_RUNNING = 60   # in kW during operation
HOTEL_LOAD_POWER_RESTING = 100    # in kW during rest

CHARGING_POWER_STATION1 = 2000  # in kW
CHARGING_POWER_STATION2 = 0     # in kW

CONNECTION_TIME = 60  # in seconds (time to connect)
SPEED_THRESHOLD = 0.2  # in knots

# Efficiency Factors
EFF_EL_MOTOR = 0.98
EFF_CABLE = 0.992
EFF_DRIVE = 0.98
EFF_BATTERY_DIS = 0.97
EFF_SEA_MARGIN = 0.95
EL_MISC_LOSS = 0.9

EL_LOSS_PROP = EFF_EL_MOTOR * EFF_CABLE * EFF_DRIVE * EFF_BATTERY_DIS * EFF_SEA_MARGIN * EL_MISC_LOSS
EL_LOSS_HOTEL = EFF_CABLE * EFF_BATTERY_DIS * EL_MISC_LOSS

# Delta speed thresholds and multiplication factors
DELTA_SPEED_THRESHOLDS = [2, 4]  # in knots
MULTIPLICATION_FACTORS = [2, 2.5]

# [haradhan] Start

def setup_script_logger(log_file_name, log_directory, session_id=None):
    log_path = os.path.join(log_directory, log_file_name)
    logger_name = "script_logger" if session_id is None else f"script_logger_{session_id}"    
  
    logger = d_logging.getLogger(logger_name)    
   
    logger.setLevel(d_logging.DEBUG)    
   
    file_handler = d_logging.FileHandler(log_path)
    file_handler.setLevel(d_logging.DEBUG)  
    formatter = d_logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console = d_logging.StreamHandler()
    console.setLevel(d_logging.INFO)
    console.setFormatter(d_logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)
    
    return logger

class DjangoCacheBackend(requests_cache.backends.BaseCache):
    def __init__(self, cache_alias="default", session_id=None, **kwargs):
        super().__init__(**kwargs)
        self.cache = caches[cache_alias]
        self.session_id = session_id or get_random_string(12)
        
    def _get_key(self, key):
        return f"{self.session_id}:{key}"

    def get(self, key):
        return self.cache.get(self._get_key(key))
    
    def set(self, key, value, expire_after=None):
        timeout = expire_after if expire_after is not None else 3600
        self.cache.set(self._get_key(key), value, timeout=timeout)

    def delete(self, key):
        self.cache.delete(self._get_key(key))
        
    def clear(self):
        pass   


def get_user_cache(session_id, output_dir): 
    
    backend = DjangoCacheBackend(session_id=session_id)
    session = requests_cache.CachedSession(backend=backend, expire_after=3600)

    return session

def get_openmeteo_client(session_id, output_dir):
    
    user_cache = get_user_cache(session_id, output_dir)
    retry_session = retry(user_cache, retries=7, backoff_factor=0.2)
    
    return openmeteo_requests.Client(session=retry_session)

def cleanup_logging(session_id=None):
    logger_name = "script_logger" if session_id is None else f"script_logger_{session_id}"
    logger = d_logging.getLogger(logger_name)
    logger.info("Processing complete. Cleaning up log handlers.")    

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

# [haradhan] End


# =============================================================================
# API and Caching Setup
# =============================================================================
CACHE_SESSION = requests_cache.CachedSession('.cache', expire_after=3600)
RETRY_SESSION = retry(CACHE_SESSION, retries=5, backoff_factor=0.2)
OPENMETEO_CLIENT = openmeteo_requests.Client(session=RETRY_SESSION)


class Analysis: # [haradhan] === converted to class to work efficiently
    
    # [haradhan] Start    
    def __init__(self, session_id = None, media_path = None):
        self.session_id = session_id
        self.media_path = media_path     
        self.logging = None
    # [haradhan] End    


    # =============================================================================
    # Power Calculation Functions
    # =============================================================================
    @staticmethod
    def power_calc(speed: float) -> float:
        """
        Calculate power based on speed using an exponential model.
        """
        MAX_SPEED = 12  # in knots
        PROPELLER_SHAFT_POWER = 1114  # in kW
        EXPONENT = 2.5
        return PROPELLER_SHAFT_POWER * (speed / MAX_SPEED) ** EXPONENT

    @staticmethod
    def power_calc_interpolated(speed: float) -> float:
        """
        Calculate power by interpolating between predefined speed and power values.
        """
        speed_values = [2.43, 3.64, 4.86, 6.07, 7.29, 8.5, 9.25, 10, 12]  # in knots
        power_values = [1.125, 17.765, 41, 81.432, 156.375, 304.152, 462.672, 708.292, 1114]  # in kW
        return np.interp(speed, speed_values, power_values)


    # =============================================================================
    # Data Handling Functions
    # =============================================================================
   
    def read_data(self, file_path: str, timezone: int) -> gpd.GeoDataFrame:
        """
        Read GPS data from a GeoJSON file and adjust the timezone.
        """
        try:
            traffic_data = gpd.read_file(file_path)
        except Exception as e:
            # [haradhan] accessing self modified logger for this session
            self.logging.exception(f"Could not read file: {e}")
            
            # [haradhan] Start
            # can crush the web server
            # sys.exit(1) 
            return {}
            # [haradhan] End
            
        traffic_data['msgtime'] = pd.to_datetime(traffic_data['msgtime']) + timedelta(hours=timezone)
        traffic_data = traffic_data.sort_values('msgtime').reset_index(drop=True)
        traffic_data['speedOverGround'] = traffic_data['speedOverGround'].fillna(0)
        traffic_data = traffic_data[traffic_data['speedOverGround'] >= 0]
        return traffic_data

    @staticmethod
    def process_trips(traffic_data: gpd.GeoDataFrame, threshold: float) -> list:
        """
        Divide GPS data into separate trips based on a given speed threshold.
        """
        speed_data = traffic_data['speedOverGround'].reset_index(drop=True)
        time_data = traffic_data['msgtime'].reset_index(drop=True)
        lon_data = traffic_data['lon'].reset_index(drop=True)
        lat_data = traffic_data['lat'].reset_index(drop=True)
        heading_data = traffic_data['trueHeading'].reset_index(drop=True)

        time_diff_hours = time_data.diff().dt.total_seconds() / 3600
        time_diff_hours = time_diff_hours.fillna(0)

        trips = []
        current_trip = []
        for i, (speed, timestamp, lon, lat, true_heading) in enumerate(zip(
                speed_data, time_data, lon_data, lat_data, heading_data)):
            if speed > threshold:
                current_trip.append((i, speed, timestamp, time_diff_hours.iloc[i],
                                    time_data.diff().dt.total_seconds().iloc[i], lon, lat, true_heading))
            else:
                if current_trip:
                    trips.append(current_trip)
                    current_trip = []
        if current_trip:
            trips.append(current_trip)
        return trips


    # =============================================================================
    # Weather and Ocean Current Data Functions
    # =============================================================================
    # [haradhan] Start
    
    # passing session_id, output_dir
    # def get_weather_data(lat: float, lon: float, date_time: pd.Timestamp) -> tuple:
   
    def get_weather_data(self, lat: float, lon: float, date_time: pd.Timestamp, session_id=None, output_dir = None) -> tuple:
    # [haradhan] End
    
        """
        Retrieve wind speed and wind direction from the Open-Meteo API.
        """
        date_str = date_time.strftime('%Y-%m-%d')
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["windspeed_10m", "winddirection_10m"],
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "UTC"
        }
        try:
            # [haradhan] Start
            
            if session_id is not None and output_dir is not None:            
                omt = get_openmeteo_client(session_id, output_dir)
                responses = omt.weather_api(url, params=params)
            else:
                responses = OPENMETEO_CLIENT.weather_api(url, params=params)    
            # responses = OPENMETEO_CLIENT.weather_api(url, params=params)
            
            # [haradhan] End     
            
            
            response = responses[0]
            hourly = response.Hourly()
            times = pd.to_datetime(hourly.Time(), unit='s', utc=True)
            wind_speed = hourly.Variables(0).ValuesAsNumpy()
            wind_direction = hourly.Variables(1).ValuesAsNumpy()
            weather_df = pd.DataFrame({
                'time': times,
                'wind_speed_10m': wind_speed / 3.6,  # Convert km/h to m/s
                'wind_direction_10m': wind_direction
            })
            weather_df.fillna(0, inplace=True)
            idx = (weather_df['time'] - date_time).abs().idxmin()
            weather_info = weather_df.iloc[idx]
            return weather_info['wind_speed_10m'], weather_info['wind_direction_10m']
        except Exception as e:
            # [haradhan] accessing self modified logger for this session
            self.logging.exception(f"Error retrieving weather data: {e}")
            return 0.0, 0.0
        
    # [haradhan] Start
    # def get_wave_data(lat: float, lon: float, date_time: pd.Timestamp) -> tuple:
   
    def get_wave_data(self, lat: float, lon: float, date_time: pd.Timestamp, session_id=None, output_dir=None) -> tuple:
    # [haradhan] End
        
        """
        Retrieve wave height, wave direction, ocean current speed, and ocean current direction from the Open-Meteo Marine API.
        """
        date_str = date_time.strftime('%Y-%m-%d')
        url = "https://marine-api.open-meteo.com/v1/marine"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["wave_height", "wave_direction", "ocean_current_velocity", "ocean_current_direction"],
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "UTC"
        }
        try:
            # [haradhan] Start
            
            if session_id is not None and output_dir is not None:            
                omt = get_openmeteo_client(session_id, output_dir)
                responses = omt.weather_api(url, params=params)
            else:
                responses = OPENMETEO_CLIENT.weather_api(url, params=params)        
            # responses = OPENMETEO_CLIENT.weather_api(url, params=params)
                    
            # [haradhan] End
            
            response = responses[0]
            hourly = response.Hourly()
            times = pd.to_datetime(hourly.Time(), unit='s', utc=True)
            wave_height = hourly.Variables(0).ValuesAsNumpy()
            wave_direction = hourly.Variables(1).ValuesAsNumpy()
            ocean_current_velocity = hourly.Variables(2).ValuesAsNumpy()
            ocean_current_direction = hourly.Variables(3).ValuesAsNumpy()
            wave_df = pd.DataFrame({
                'time': times,
                'wave_height': wave_height,
                'wave_direction': wave_direction,
                'ocean_current_velocity': ocean_current_velocity,
                'ocean_current_direction': ocean_current_direction
            })
            wave_df.fillna(0, inplace=True)
            idx = (wave_df['time'] - date_time).abs().idxmin()
            wave_info = wave_df.iloc[idx]
            return (wave_info['wave_height'], wave_info['wave_direction'],
                    wave_info['ocean_current_velocity'], wave_info['ocean_current_direction'])
        except Exception as e:
            # [haradhan] accessing self modified logger for this session
            self.logging.exception(f"Error retrieving wave data: {e}")
            return 0.0, 0.0, 0.0, 0.0


    # =============================================================================
    # Energy Consumption Calculation Function
    # =============================================================================
    # [haradhan] Start
    
    # def calculate_energy_consumption(trips: list) -> dict:   
    def calculate_energy_consumption(self, trips: list, session_id=None, output_dir=None) -> dict:
        
    # [haradhan] End  
        """
        Calculate energy consumption and battery level for each trip.
        """
        battery_level = BATTERY_CAPACITY
        battery_levels = [battery_level]
        time_stamps = []
        trip_energy_data = []
        trip_speed_data = []
        trip_weather_data = []
        for i, trip in enumerate(trips, 1):
            total_energy = 0
            total_speed = 0
            num_points = 0

            time_points = []
            energy_points = []
            speed_points = []
            wind_speed_points = []
            wind_direction_points = []
            wave_height_points = []
            wave_direction_points = []
            ocean_current_velocity_points = []
            ocean_current_direction_points = []
            true_heading_points = []

            start_time_trip = trip[0][2]
            end_time_trip = trip[-1][2]
            trip_duration_seconds = (end_time_trip - start_time_trip).total_seconds()
            trip_duration_minutes = int(trip_duration_seconds // 60)
            trip_duration_seconds_remaining = int(trip_duration_seconds % 60)
            
            # [haradhan] accessing self modified logger for this session
            self.logging.info(f"Trip {i} (Duration: {trip_duration_minutes}:{trip_duration_seconds_remaining:02d} mm:ss):")

            previous_speed = 0
            for point in trip:
                idx, speed, timestamp, delta_time_hour, delta_time_sec, lon, lat, true_heading = point
                delta_speed = speed - previous_speed

                # Calculate power based on speed using chosen method
                if POWER_CALC_OPTION:
                    power = Analysis.power_calc_interpolated(speed)
                else:
                    power = Analysis.power_calc(speed) / EL_LOSS_PROP

                # Adjust power based on increase in speed
                if delta_speed > 0:
                    if delta_speed > DELTA_SPEED_THRESHOLDS[1]:
                        power *= MULTIPLICATION_FACTORS[1]
                    elif delta_speed > DELTA_SPEED_THRESHOLDS[0]:
                        power *= MULTIPLICATION_FACTORS[0]

                energy = power * delta_time_hour
                hotel_energy = (HOTEL_LOAD_POWER_RUNNING * delta_time_hour) / EL_LOSS_HOTEL
                total_energy_segment = energy + hotel_energy
                total_energy += total_energy_segment

                time_points.append(timestamp)
                energy_points.append(total_energy_segment)
                speed_points.append(speed)
                true_heading_points.append(true_heading)

                total_speed += speed
                num_points += 1

                # Retrieve weather and wave data for the current timestamp and location
                # wind_speed, wind_direction = get_weather_data(lat, lon, timestamp)
                # wave_height, wave_direction, ocean_current_velocity, ocean_current_direction = get_wave_data(lat, lon, timestamp)
                
                # [haradhan] Start
                
                if session_id is not None and output_dir is not None:
                    wind_speed, wind_direction = self.get_weather_data(lat, lon, timestamp, session_id, output_dir)
                    wave_height, wave_direction, ocean_current_velocity, ocean_current_direction = self.get_wave_data(lat, lon, timestamp, session_id, output_dir)
                else:
                    # original script
                    wind_speed, wind_direction = self.get_weather_data(lat, lon, timestamp)
                    wave_height, wave_direction, ocean_current_velocity, ocean_current_direction = self.get_wave_data(lat, lon, timestamp)
                    
                # [haradhan] End
                    
                    
                

                wind_speed_points.append(wind_speed)
                wind_direction_points.append(wind_direction)
                wave_height_points.append(wave_height)
                wave_direction_points.append(wave_direction)
                ocean_current_velocity_points.append(ocean_current_velocity)
                ocean_current_direction_points.append(ocean_current_direction)

                formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                # [haradhan] accessing self modified logger for this session, extra character (arrow) from this line ha been removed
                self.logging.info(f"Time: {formatted_timestamp}, Speed: {speed:.2f} knots, Speed: {delta_speed:.2f} knots, Power: {power:.2f} kW, Time: {delta_time_sec:.2f} sec, Energy: {total_energy_segment:.4f} kWh")
                previous_speed = speed

            trip_energy_data.append({
                'trip_number': i,
                'time_points': time_points,
                'energy_points': energy_points
            })
            trip_speed_data.append({
                'trip_number': i,
                'time_points': time_points,
                'speed_points': speed_points
            })
            trip_weather_data.append({
                'trip_number': i,
                'time_points': time_points,
                'wind_speed_points': wind_speed_points,
                'wind_direction_points': wind_direction_points,
                'wave_height_points': wave_height_points,
                'wave_direction_points': wave_direction_points,
                'ocean_current_velocity_points': ocean_current_velocity_points,
                'ocean_current_direction_points': ocean_current_direction_points,
                'true_heading_points': true_heading_points
            })

            average_speed = total_speed / num_points if num_points > 0 else 0
            
            # [haradhan] accessing self modified logger for this session
            self.logging.info(f"Total energy consumption for Trip {i}: {total_energy:.4f} kWh (including hotel load)")          
            self.logging.info(f"Average speed for Trip {i}: {average_speed:.2f} knots")
            
            battery_level = max(0, battery_level - total_energy)
            if battery_level == 0:
                # [haradhan] accessing self modified logger for this session
                self.logging.warning("Warning: Battery is empty!")
            battery_levels.append(battery_level)
            time_stamps.append(trip[-1][2])
            
            # Charging between trips
            if i < len(trips):
                end_time_current_trip = trips[i - 1][-1][2]
                start_time_next_trip = trips[i][0][2]
                rest_time_seconds = max((start_time_next_trip - end_time_current_trip).total_seconds() - CONNECTION_TIME, 0)
                if rest_time_seconds > 0:
                    charging_power = CHARGING_POWER_STATION1 if i % 2 == 1 else CHARGING_POWER_STATION2
                    charging_time_hours = rest_time_seconds / 3600
                    hotel_energy_rest = (HOTEL_LOAD_POWER_RESTING * charging_time_hours) / EL_LOSS_HOTEL
                    energy_charged = charging_power * charging_time_hours * EL_LOSS_PROP
                    battery_level = min(BATTERY_CAPACITY, battery_level + energy_charged - hotel_energy_rest)
                    
                    # [haradhan] accessing self modified logger for this session
                    self.logging.info(f"Charging at station {1 if i % 2 == 1 else 2} for {charging_time_hours:.2f} hours at {charging_power} kW, charged {energy_charged:.2f} kWh")                    
                    self.logging.info(f"Hotel load consumed {hotel_energy_rest:.2f} kWh during rest period")
                else:
                    hotel_energy_rest = (HOTEL_LOAD_POWER_RESTING * (rest_time_seconds / 3600)) / EL_LOSS_HOTEL
                    battery_level = max(0, battery_level - hotel_energy_rest)
                    if battery_level == 0:
                        
                        # [haradhan] accessing self modified logger for this session
                        self.logging.warning("Warning: Battery is empty during rest period!")
                        
                battery_levels.append(battery_level)
                time_stamps.append(start_time_next_trip)
                
                # [haradhan] accessing self modified logger for this session
                self.logging.info(f"Battery level after Trip {i}: {battery_level:.2f} kWh\n")
                
        return {
            'battery_levels': battery_levels,
            'time_stamps': time_stamps,
            'trip_energy_data': trip_energy_data,
            'trip_speed_data': trip_speed_data,
            'trip_weather_data': trip_weather_data,
        }


    # =============================================================================
    # Plotting Functions
    # =============================================================================
    
    def plot_battery_levels(self, time_stamps: list, battery_levels: list, output_dir: str):
        """
        Plot battery levels over time and save the chart as a PNG file.
        """
        # [haradhan] Start
        
        # plt.figure(figsize=(12, 6))
        fig = plt.figure(figsize=(12, 6))
        # [haradhan] End
        
        plt.plot(time_stamps, battery_levels[1:], marker='o')
        plt.gcf().autofmt_xdate()
        plt.xlabel('Time')
        plt.ylabel('Battery Level (kWh)')
        plt.title('Battery Level Over Time')
        plt.grid(True)
        output_path = os.path.join(output_dir, "battery_levels.png")
        plt.savefig(output_path)
        
        # [haradhan] Start
        # plt.close()
        plt.close(fig)
        # [haradhan] End
        
        # [haradhan] accessing self modified logger for this session
        self.logging.info(f"Battery level plot saved as {output_path}")
        
   
    def plot_trip_on_map(self, trip_number: int, trips: list, traffic_data: gpd.GeoDataFrame, results: dict, output_dir: str):
        """
        Plot a specific trip on a map with weather and ocean current data and save it as an HTML file.
        """
        if trip_number < 1 or trip_number > len(trips):
            
            # [haradhan] accessing self modified logger for this session
            self.logging.error(f"Trip {trip_number} is out of range.")
            
            return
        
        trip_indices = [pt[0] for pt in trips[trip_number - 1]]
        trip_data = traffic_data.iloc[trip_indices]
        route = LineString(zip(trip_data['lon'], trip_data['lat']))
        start_location = [trip_data.iloc[0]['lat'], trip_data.iloc[0]['lon']]
        m = folium.Map(location=start_location, zoom_start=12)
        folium.GeoJson(route, name=f'Trip {trip_number}').add_to(m)
        
        trip_weather = results['trip_weather_data'][trip_number - 1]
        wind_speed_points = trip_weather['wind_speed_points']
        wind_direction_points = trip_weather['wind_direction_points']
        wave_height_points = trip_weather['wave_height_points']
        wave_direction_points = trip_weather['wave_direction_points']
        ocean_current_velocity_points = trip_weather['ocean_current_velocity_points']
        ocean_current_direction_points = trip_weather['ocean_current_direction_points']
        
        # Boat Markers
        boat_layer = folium.FeatureGroup(name='Boat Positions', overlay=True)
        for (_, row), wind_speed, wind_direction in zip(trip_data.iterrows(), wind_speed_points, wind_direction_points):
            BoatMarker(
                location=[row['lat'], row['lon']],
                heading=row['trueHeading'],
                wind_heading=wind_direction,
                wind_speed=wind_speed,
                color='red',
                popup=f"<b>Time:</b> {row['msgtime']}<br><b>Speed:</b> {row['speedOverGround']:.2f} knots<br><b>Heading:</b> {row['trueHeading']:.2f}°"
            ).add_to(boat_layer)
        boat_layer.add_to(m)
        
        # Wave Data
        wave_layer = folium.FeatureGroup(name='Wave Data', overlay=True)
        for (_, row), wave_height, wave_direction in zip(trip_data.iterrows(), wave_height_points, wave_direction_points):
            if wave_height < 1:
                color = 'blue'
                wave_category = 'Under 1 m'
            elif wave_height < 2:
                color = 'green'
                wave_category = '1 - 2 m'
            else:
                color = 'red'
                wave_category = 'Over 2 m'
            angle_rad = math.radians(wave_direction)
            delta_lat = math.cos(angle_rad) * 0.001
            delta_lon = math.sin(angle_rad) * 0.001
            start_point = [row['lat'], row['lon']]
            end_point = [row['lat'] + delta_lat, row['lon'] + delta_lon]
            folium.PolyLine(
                locations=[start_point, end_point],
                color=color,
                weight=3,
                opacity=0.8,
                popup=f"<b>Time:</b> {row['msgtime']}<br><b>Wave Height:</b> {wave_height:.2f} m ({wave_category})<br><b>Wave Direction:</b> {wave_direction:.2f}°"
            ).add_to(wave_layer)
        wave_layer.add_to(m)
        
        # Ocean Current Data
        ocean_layer = folium.FeatureGroup(name='Ocean Current Data', overlay=True)
        for (_, row), ocean_speed, ocean_direction in zip(trip_data.iterrows(),
                                                            ocean_current_velocity_points,
                                                            ocean_current_direction_points):
            angle_rad = math.radians(ocean_direction)
            delta_lat = math.cos(angle_rad) * 0.001
            delta_lon = math.sin(angle_rad) * 0.001
            start_point = [row['lat'], row['lon']]
            end_point = [row['lat'] + delta_lat, row['lon'] + delta_lon]
            folium.PolyLine(
                locations=[start_point, end_point],
                color='orange',
                weight=3,
                opacity=0.8,
                dash_array='5,5',
                popup=f"<b>Time:</b> {row['msgtime']}<br><b>Ocean Current Speed:</b> {ocean_speed:.2f} m/s<br><b>Ocean Current Direction:</b> {ocean_direction:.2f}°"
            ).add_to(ocean_layer)
        ocean_layer.add_to(m)
        
        folium.LayerControl().add_to(m)
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; font-size:14px;">
        &nbsp;<b>Legend</b><br>
        &nbsp;<i style="background: blue; width: 30px; height: 3px; display: inline-block;"></i>&nbsp;Wave < 1 m<br>
        &nbsp;<i style="background: green; width: 30px; height: 3px; display: inline-block;"></i>&nbsp;Wave 1-2 m<br>
        &nbsp;<i style="background: red; width: 30px; height: 3px; display: inline-block;"></i>&nbsp;Wave > 2 m<br>
        &nbsp;<i style="border-bottom: 3px dashed orange; width: 30px; display: inline-block;"></i>&nbsp;Ocean Current
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        output_file = os.path.join(output_dir, f"trip_{trip_number}_map.html")
        m.save(output_file)
        
        # [haradhan] accessing self modified logger for this session
        self.logging.info(f"Trip {trip_number} map saved as {output_file}")
        

    def plot_energy_consumption(self, trip_number: int, trip_energy_data: list, output_dir: str):
        """
        Plot cumulative energy consumption for a trip and save as a PNG file.
        """
        if trip_number < 1 or trip_number > len(trip_energy_data):
            # [haradhan] accessing self modified logger for this session
            self.logging.error(f"Trip {trip_number} is out of range.")
            return
        trip_data = trip_energy_data[trip_number - 1]
        time_points = trip_data['time_points']
        energy_points = trip_data['energy_points']
        cumulative_energy = np.cumsum(energy_points)
        
        # [haradhan] Start
        # plt.figure(figsize=(12, 6))
        fig = plt.figure(figsize=(12, 6))
        # [haradhan] End
        
        
        plt.plot(time_points, cumulative_energy, marker='o')
        plt.gcf().autofmt_xdate()
        plt.xlabel('Time')
        plt.ylabel('Cumulative Energy Consumption (kWh)')
        plt.title(f'Energy Consumption for Trip {trip_number}')
        plt.grid(True)
        output_path = os.path.join(output_dir, f"trip_{trip_number}_energy.png")
        plt.savefig(output_path)
        
        # [haradhan] Start
        # plt.close()
        plt.close(fig)
        # [haradhan] End
        
        # [haradhan] accessing self modified logger for this session
        self.logging.info(f"Energy consumption plot for Trip {trip_number} saved as {output_path}")


    def plot_speed_over_time(self, trip_number: int, trip_speed_data: list, output_dir: str):
        """
        Plot speed over time for a trip and save as a PNG file.
        """
        
        if trip_number < 1 or trip_number > len(trip_speed_data):
            
            # [haradhan] accessing self modified logger for this session
            self.logging.error(f"Trip {trip_number} is out of range.")
            
            return
        
        trip_data = trip_speed_data[trip_number - 1]
        time_points = trip_data['time_points']
        speed_points = trip_data['speed_points']
        
        # [haradhan] Start 
        # plt.figure(figsize=(12, 6))
        fig = plt.figure(figsize=(12, 6))
        # [haradhan] End
        
        plt.plot(time_points, speed_points, marker='o')
        plt.gcf().autofmt_xdate()
        plt.xlabel('Time')
        plt.ylabel('Speed (knots)')
        plt.title(f'Speed Over Time for Trip {trip_number}')
        plt.grid(True)
        output_path = os.path.join(output_dir, f"trip_{trip_number}_speed.png")
        plt.savefig(output_path)
        
        # [haradhan] Start      
        # plt.close()
        plt.close(fig)    
        # [haradhan] End
        
        # [haradhan] accessing self modified logger for this session
        self.logging.info(f"Speed plot for Trip {trip_number} saved as {output_path}")


    # =============================================================================
    # Main Function.
    # =============================================================================
    
    # [haradhan] Start
    # renamed main to use in the web
    # def main()

    def run_energy_analysis(self, file_path):
    # [haradhan] End       
        

       
    
        # [haradhan] Transfered to __name__
        # file_path = input("Enter the file path for the GeoJSON file: ").strip().strip('"')        
        
        if not os.path.exists(file_path):
            print("File not found. Exiting.")
            
            # [haradhan] Start
            # can crush the web server
            # sys.exit(1)     
            return
            # [haradhan] End
            
            
            
        # [haradhan] Start
        
        # if session id making it session specific
        if self.session_id is not None and self.media_path is not None:        
            # session specific for web use      
            output_dir = os.path.join(self.media_path, f"output_{self.session_id}")
            
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)  
        
        else:    
            output_dir = "output"
        # output_dir = "output"          
            
        # [haradhan] End
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")        
        
        log_filename=os.path.join(output_dir, "output_log.log") if self.session_id is None else os.path.join(output_dir, f"output_log_{self.session_id}.log")    
        # [haradhan] accessing self modified logger for this session
        self.logging = setup_script_logger(log_filename, output_dir, self.session_id)
        
        
        # [haradhan] accessing self modified logger for this session
        self.logging.info(f"Starting processing with input file: {file_path}")
                
        
        # Read and prepare data
        # [haradhan] as script now under class so will access by self
        traffic_data = self.read_data(file_path, TIMEZONE)
        traffic_data['lon'] = traffic_data.geometry.x
        traffic_data['lat'] = traffic_data.geometry.y
        trips = Analysis.process_trips(traffic_data, SPEED_THRESHOLD)
        if not trips:
            # [haradhan] accessing self modified logger for this session
            self.logging.error("No trips found with the current speed threshold.")
            
            # [haradhan] Start
            # can crush the web server
            # sys.exit(1)
            return
            # [haradhan] End
        
        # Calculate energy consumption and battery levels
        
        # [haradhan] Start   
        if self.session_id is not None:
            results = self.calculate_energy_consumption(trips, self.session_id, output_dir)
        else:
            results = self.calculate_energy_consumption(trips)
        # results = calculate_energy_consumption(trips)    
        # [haradhan] End
            
        
        # Save battery level plot
        
        # [haradhan] as script now under class so will access by self
        self.plot_battery_levels(results['time_stamps'], results['battery_levels'], output_dir)
        
        # For each trip, save map, energy plot, and speed plot
        for i in range(1, len(trips) + 1):
            
            # [haradhan] accessing self modified logger for this session
            self.logging.info(f"Processing Trip {i}...")
            
            # [haradhan] Start 
            # as script now under class so will access by self
            self.plot_trip_on_map(i, trips, traffic_data, results, output_dir)
            self.plot_energy_consumption(i, results['trip_energy_data'], output_dir)
            self.plot_speed_over_time(i, results['trip_speed_data'], output_dir)
            # [haradhan] End
        
        # Create a map with all trips
        m = folium.Map(location=[traffic_data['lat'].mean(), traffic_data['lon'].mean()], zoom_start=12)
        num_trips = len(trips)
        colors = [
            f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'
            for r, g, b in [colorsys.hls_to_rgb(i / num_trips, 0.5, 0.9) for i in range(num_trips)]
        ]
        for i, trip in enumerate(trips, 1):
            trip_indices = [pt[0] for pt in trip]
            trip_data = traffic_data.iloc[trip_indices]
            route = LineString(zip(trip_data['lon'], trip_data['lat']))
            color = colors[i - 1]
            folium.GeoJson(
                route,
                name=f'Trip {i}',
                style_function=lambda feature, color=color: {'color': color, 'weight': 3},
                tooltip=f'Trip {i}'
            ).add_to(m)
        folium.LayerControl().add_to(m)
        all_trips_map = os.path.join(output_dir, "all_trips_map.html")
        m.save(all_trips_map)
        
        # [haradhan] accessing self modified logger for this session
        self.logging.info(f"Combined map with all trips saved as {all_trips_map}")
        
        
        # [haradhan] Start
        if self.session_id and self.media_path:    
            cleanup_logging(self.session_id)    
            return output_dir    
        # [haradhan] End

if __name__ == "__main__":
    # Ask the user for the file path (quotes around the path are allowed)
    # [haradhan] Start
    # Bringing here from perevious main()
    file_path = input("Enter the file path for the GeoJSON file: ").strip().strip('"')
    # [haradhan] End
    
    # SAFE TO INACTIVE IN WEB MODE
    # If want to run file independently just uncomment below
    # Analysis().run_energy_analysis(file_path)
