# aqi.py
import numpy as np
import pandas as pd

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50, "Good"),
    (12.1, 35.4, 51, 100, "Moderate"),
    (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups"),
    (55.5, 150.4, 151, 200, "Unhealthy"),
    (150.5, 250.4, 201, 300, "Very Unhealthy"),
    (250.5, 500.4, 301, 500, "Hazardous"),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50, "Good"),
    (55, 154, 51, 100, "Moderate"),
    (155, 254, 101, 150, "Unhealthy for Sensitive Groups"),
    (255, 354, 151, 200, "Unhealthy"),
    (355, 424, 201, 300, "Very Unhealthy"),
    (425, 604, 301, 500, "Hazardous"),
]

def _calc_sub_index(conc, breakpoints):
    for Clow, Chigh, Ilow, Ihigh, cat in breakpoints:
        if conc >= Clow and conc <= Chigh:
            aqi = ((Ihigh - Ilow) / (Chigh - Clow)) * (conc - Clow) + Ilow
            return round(aqi), cat
    if conc < breakpoints[0][0]:
        return breakpoints[0][2], breakpoints[0][4]
    return breakpoints[-1][3], breakpoints[-1][4]

def calculate_aqi(row):
    pm25_col = None
    pm10_col = None
    for c in row.index:
        if c.lower().replace(".", "").replace(" ", "") in ["pm25", "pm2_5", "pm2.5"]:
            pm25_col = c
        if c.lower().replace(".", "").replace(" ", "") in ["pm10"]:
            pm10_col = c

    sub_indices = []
    categories = []
    if pm25_col and not pd.isna(row[pm25_col]):
        val = row[pm25_col]
        aqi, cat = _calc_sub_index(val, PM25_BREAKPOINTS)
        sub_indices.append(aqi); categories.append(cat)
    if pm10_col and not pd.isna(row[pm10_col]):
        val = row[pm10_col]
        aqi, cat = _calc_sub_index(val, PM10_BREAKPOINTS)
        sub_indices.append(aqi); categories.append(cat)

    if not sub_indices:
        return None, "Unknown"
    max_idx = max(sub_indices)
    cat = categories[sub_indices.index(max_idx)]
    return max_idx, cat
