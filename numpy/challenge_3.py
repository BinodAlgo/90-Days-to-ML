# Flash Flood Detector 
""" 
Problem Statement: You are monitoring a river sensor in Yamanashi. You have a matrix 
representing data of 7 days, with 24 hours of daily readings.

Anomalies: Calculate the global mean and global standard deviation of the entire week.

Trigger the Alarm: Find the exact indices (Day #, Hour #) where the water level was more than 2 standard deviations above the global mean.


"""
import numpy as np 
river_sensor_data = np.random.normal(3.0,1.0,(7,24))
print(river_sensor_data)

# Maximum water levels each day
max_water_levels = np.max(river_sensor_data, axis = 1)
print(f"Maximum water levels each day: {max_water_levels}")

# Identify the hour with highest water level
highest_water_level_hour = np.argmax(river_sensor_data, axis = 1)
print(f"The hour with the highest water level is: {highest_water_level_hour}")

# Check for danger levels
# Create a boolean mask where values exceed 4.0 (danger zone)
danger_zone = max_water_levels > 4.0
print(f"Danger zone: {danger_zone}")

# Calcualte global mean and standar deviation
mean = np.mean(river_sensor_data)
std = np.std(river_sensor_data)
print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")

# Identify the indices where water level was more than 2 standard deviations above the global mean
anomalies = np.argwhere(river_sensor_data > mean + 2 * std)
print(f"Anomalies found on days: {anomalies}")

