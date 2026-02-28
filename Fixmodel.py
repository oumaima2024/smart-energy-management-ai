# Quick script to regenerate feature_cols.pkl and scaler_dl.pkl
# Run this once: python generate_missing_files.py

import numpy as np
import pandas as pd
import joblib
import glob
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load CSV ──────────────────────────────────────────────────────────────────
csv_files = glob.glob(os.path.join(BASE, "data\Energy_consumption.csv"))
if not csv_files:
    print("ERROR: No CSV file found in this folder!")
    exit(1)

df = pd.read_csv(csv_files[0])
print(f"Loaded {len(df)} rows from {os.path.basename(csv_files[0])}")

# ── Parse timestamp ───────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

df['Hour']       = df['Timestamp'].dt.hour
df['Day']        = df['Timestamp'].dt.day
df['Month']      = df['Timestamp'].dt.month
df['DayOfYear']  = df['Timestamp'].dt.dayofyear
df['WeekOfYear'] = df['Timestamp'].dt.isocalendar().week.astype(int)

le = LabelEncoder()
df['HVACUsage']     = le.fit_transform(df['HVACUsage'])
df['LightingUsage'] = le.fit_transform(df['LightingUsage'])
df['Holiday']       = le.fit_transform(df['Holiday'])

day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['DayOfWeek'] = pd.Categorical(df['DayOfWeek'], categories=day_order, ordered=True).codes + 1

# ── Outlier removal ───────────────────────────────────────────────────────────
num_cols = ['Temperature','Humidity','SquareFootage','Occupancy','RenewableEnergy','EnergyConsumption']
Q1, Q3 = df[num_cols].quantile(0.25), df[num_cols].quantile(0.75)
IQR    = Q3 - Q1
mask   = ((df[num_cols] < (Q1 - 1.5*IQR)) | (df[num_cols] > (Q3 + 1.5*IQR))).any(axis=1)
df     = df[~mask].copy().reset_index(drop=True)
print(f"After cleaning: {len(df)} rows")

TARGET = 'EnergyConsumption'

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(d):
    d = d.copy()
    d['Hour_sin']  = np.sin(2*np.pi*d['Hour']/24)
    d['Hour_cos']  = np.cos(2*np.pi*d['Hour']/24)
    d['Day_sin']   = np.sin(2*np.pi*d['DayOfWeek']/7)
    d['Day_cos']   = np.cos(2*np.pi*d['DayOfWeek']/7)
    d['Month_sin'] = np.sin(2*np.pi*d['Month']/12)
    d['Month_cos'] = np.cos(2*np.pi*d['Month']/12)

    for lag in [1,2,3,6,12,24,48]:
        d[f'lag_{lag}'] = d[TARGET].shift(lag)
    for w in [3,6,12,24]:
        d[f'roll_mean_{w}'] = d[TARGET].shift(1).rolling(w).mean()
        d[f'roll_std_{w}']  = d[TARGET].shift(1).rolling(w).std()

    d['diff_1']  = d[TARGET].diff(1)
    d['diff_24'] = d[TARGET].diff(24)

    d['Temp_x_HVAC']      = d['Temperature'] * d['HVACUsage']
    d['Temp_x_Occupancy'] = d['Temperature'] * d['Occupancy']
    d['Temp_x_Humidity']  = d['Temperature'] * d['Humidity']
    d['HVAC_x_Lighting']  = d['HVACUsage']   * d['LightingUsage']
    d['Occ_x_Lighting']   = d['Occupancy']   * d['LightingUsage']

    d['Temp_sq']  = d['Temperature']**2
    d['Humid_sq'] = d['Humidity']**2

    d['OccDensity'] = d['Occupancy'] / (d['SquareFootage'] + 1)
    d['RenewRatio'] = d['RenewableEnergy'] / (d['Temperature'] + 1)
    d['IsWeekend']  = (d['DayOfWeek'] >= 6).astype(int)
    d['PeakHour']   = d['Hour'].apply(lambda h: 1 if (8<=h<=10 or 17<=h<=20) else 0)
    return d.dropna()

df_feat = engineer_features(df)

FEAT_COLS = [c for c in df_feat.columns if c not in [TARGET, 'Timestamp']]
print(f"Feature columns: {len(FEAT_COLS)}")

# ── Save feature_cols.pkl ─────────────────────────────────────────────────────
joblib.dump(FEAT_COLS, os.path.join(BASE, 'feature_cols.pkl'))
print("Saved: feature_cols.pkl")

# ── Save scaler_dl.pkl ────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8
split_idx   = int(len(df_feat) * TRAIN_RATIO)
ts_full     = df_feat[TARGET].values.reshape(-1, 1)

scaler_dl = MinMaxScaler()
scaler_dl.fit(ts_full[:split_idx])
joblib.dump(scaler_dl, os.path.join(BASE, 'scaler_dl.pkl'))
print("Saved: scaler_dl.pkl")

print("\nDone! Restart app.py now.")
print("You should see: ML model OK + Features OK + Scaler OK")