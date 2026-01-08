#!/usr/bin/env python3
import numpy as np
import sys

# Load data
data = np.load('/home/clear/recorded_data/episode_1/xela_data.npy', allow_pickle=True)

print(f"Total frames: {len(data)}")
print(f"Duration: {data[-1]['timestamp'] - data[0]['timestamp']:.2f}s")
print(f"\nFirst frame:")

for i, sensor in enumerate(data[0]['sensors']):
    print(f"  Sensor {i}: {len(sensor['taxels'])} taxels, {len(sensor['forces'])} forces")
    print(f"    First taxel: {sensor['taxels'][0]}")
    if sensor['forces']:
        print(f"    First force: {sensor['forces'][0]}")