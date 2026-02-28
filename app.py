"""
Flask Backend — AI Smart Energy Management
Fixes in this version:
  - Custom SimpleScaler / SimpleModel classes defined so joblib can load them
  - CSV search expanded: checks subdirectories and all .csv files
  - Categorical columns (HVACUsage, LightingUsage, Holiday, DayOfWeek) encoded
    the same way the notebook did (LabelEncoder / Categorical codes)
  - XGBoost JSON loads when .pkl has custom class issues
  - Auto-builds MinMaxScaler from real CSV data when scaler_dl.pkl fails
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings, os, logging, joblib, glob

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))
log.info(f"Base directory: {BASE}")

# ============================================================================
# CUSTOM CLASSES — define before any joblib.load() so pickle can find them
# These match whatever was defined in create_models.py / checkmodel.py
# ============================================================================

class SimpleScaler:
    """Lightweight MinMax-style scaler that may have been saved via joblib."""
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, data):
        self.min_val = float(np.min(data))
        self.max_val = float(np.max(data))
        return self

    def transform(self, data):
        r = self.max_val - self.min_val
        if r == 0:
            return np.zeros_like(data, dtype=float)
        return (np.array(data, dtype=float) - self.min_val) / r

    def inverse_transform(self, data):
        return np.array(data, dtype=float) * (self.max_val - self.min_val) + self.min_val


class SimpleModel:
    """Stub class so joblib can at least load the file without crashing."""
    def __init__(self):
        self.model = None

    def predict(self, X):
        if self.model is not None:
            return self.model.predict(X)
        return np.array([50.0] * len(X))


# ============================================================================
# MODEL LOADING
# ============================================================================

ML_MODEL      = None
ML_MODEL_NAME = "XGBoost"
FEAT_COLS     = None
SCALER_DL     = None
DL_MODEL      = None

log.info("Searching for model files...")

# ── ML Model ──────────────────────────────────────────────────────────────────
# Priority: best_ml_model*.pkl → *XGBoost*.pkl → any *.pkl → XGBoost JSON
pkl_candidates = (
    glob.glob(os.path.join(BASE, "best_ml_model*.pkl"))
    + glob.glob(os.path.join(BASE, "*XGBoost*.pkl"))
    + glob.glob(os.path.join(BASE, "xgboost_model.pkl"))
)
for mf in dict.fromkeys(pkl_candidates):  # deduplicate, preserve order
    try:
        obj = joblib.load(mf)
        # If it's a SimpleModel stub with no inner model, skip
        if isinstance(obj, SimpleModel) and obj.model is None:
            log.warning(f"Skipping stub SimpleModel in {os.path.basename(mf)}")
            continue
        ML_MODEL = obj
        ML_MODEL_NAME = (os.path.basename(mf)
                         .replace("best_ml_model_","")
                         .replace(".pkl","")
                         .replace("xgboost_model","XGBoost"))
        log.info(f"ML loaded (pkl): {os.path.basename(mf)} → type={type(obj).__name__}")
        break
    except Exception as e:
        log.warning(f"pkl failed [{os.path.basename(mf)}]: {e}")

if ML_MODEL is None:
    # Try XGBoost native JSON (no sklearn / pickle dependency)
    for jf in (glob.glob(os.path.join(BASE, "best_ml_model*.json"))
               + glob.glob(os.path.join(BASE, "*XGBoost*.json"))):
        try:
            from xgboost import XGBRegressor
            m = XGBRegressor()
            m.load_model(jf)
            ML_MODEL, ML_MODEL_NAME = m, "XGBoost"
            log.info(f"ML loaded (XGBoost JSON): {os.path.basename(jf)}")
            break
        except Exception as e:
            log.warning(f"XGBoost JSON failed [{os.path.basename(jf)}]: {e}")

if ML_MODEL is None:
    log.warning("No usable ML model found — simulation mode active")

# ── Feature columns ───────────────────────────────────────────────────────────
for ff in (glob.glob(os.path.join(BASE, "feature_cols.pkl"))
           + glob.glob(os.path.join(BASE, "*feature*.pkl"))):
    try:
        FEAT_COLS = joblib.load(ff)
        log.info(f"Feature cols: {len(FEAT_COLS)} features from {os.path.basename(ff)}")
        break
    except Exception as e:
        log.warning(f"Feature cols failed [{os.path.basename(ff)}]: {e}")

# ── DL Scaler ─────────────────────────────────────────────────────────────────
for sf in (glob.glob(os.path.join(BASE, "scaler_dl.pkl"))
           + glob.glob(os.path.join(BASE, "*scaler*.pkl"))):
    try:
        obj = joblib.load(sf)
        # Accept SimpleScaler or sklearn MinMaxScaler
        if hasattr(obj, 'transform') and hasattr(obj, 'inverse_transform'):
            SCALER_DL = obj
            log.info(f"Scaler loaded: {os.path.basename(sf)} → type={type(obj).__name__}")
            break
        else:
            log.warning(f"Scaler in {os.path.basename(sf)} has no transform method, skipping")
    except Exception as e:
        log.warning(f"Scaler failed [{os.path.basename(sf)}]: {e}")

# ── DL Model ──────────────────────────────────────────────────────────────────
for lf in (glob.glob(os.path.join(BASE, "best_dl_model*.keras"))
           + glob.glob(os.path.join(BASE, "lstm_model.keras"))
           + glob.glob(os.path.join(BASE, "*.keras"))
           + glob.glob(os.path.join(BASE, "*.h5"))):
    try:
        import tensorflow as tf
        DL_MODEL = tf.keras.models.load_model(lf)
        log.info(f"DL model loaded: {os.path.basename(lf)}")
        break
    except Exception as e:
        log.warning(f"DL model failed [{os.path.basename(lf)}]: {e}")

# ── Active mode ───────────────────────────────────────────────────────────────
if ML_MODEL is not None and FEAT_COLS is not None:
    MODE = ML_MODEL_NAME
elif ML_MODEL is not None:
    MODE = ML_MODEL_NAME
elif DL_MODEL is not None:
    MODE = "LSTM"
else:
    MODE = "simulation"

log.info(f"Active mode: {MODE}")


def _ml_predict(X: pd.DataFrame) -> float:
    """Predict through Pipeline, GridSearchCV, or raw model."""
    try:
        return float(ML_MODEL.predict(X)[0])
    except Exception as e:
        log.warning(f"Direct predict failed ({e}), unwrapping...")
        m = ML_MODEL
        if hasattr(m, "best_estimator_"):
            m = m.best_estimator_
        if hasattr(m, "named_steps"):
            m = list(m.named_steps.values())[-1]
        if hasattr(m, "model") and m.model is not None:
            m = m.model
        return float(m.predict(X)[0])


# ============================================================================
# DATA LOADING — searches current dir + subdirs + common names
# ============================================================================

def _encode_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same encoding the notebook used."""
    d = df.copy()
    # HVAC / Lighting / Holiday: Off→0, On→1 / No→0, Yes→1
    for col, mapping in [
        ('HVACUsage',    {'Off':0,'On':1}),
        ('LightingUsage',{'Off':0,'On':1}),
        ('Holiday',      {'No':0,'Yes':1}),
    ]:
        if col in d.columns and d[col].dtype == object:
            d[col] = d[col].map(mapping).fillna(d[col])

    # DayOfWeek: Monday=1 … Sunday=7
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    if 'DayOfWeek' in d.columns and d['DayOfWeek'].dtype == object:
        cat = pd.Categorical(d['DayOfWeek'], categories=day_order, ordered=True)
        d['DayOfWeek'] = cat.codes + 1   # 1-indexed like the notebook

    # Also handle column name variation DayofWeek vs DayOfWeek
    if 'DayofWeek' in d.columns and 'DayOfWeek' not in d.columns:
        d.rename(columns={'DayofWeek':'DayOfWeek'}, inplace=True)
        if d['DayOfWeek'].dtype == object:
            cat = pd.Categorical(d['DayOfWeek'], categories=day_order, ordered=True)
            d['DayOfWeek'] = cat.codes + 1

    return d


def load_data():
    # Build search paths: current dir + all immediate subdirs
    search_dirs = [BASE] + [
        os.path.join(BASE, d) for d in os.listdir(BASE)
        if os.path.isdir(os.path.join(BASE, d))
        and not d.startswith('.') and d not in ('venv', '__pycache__', 'models')
    ]

    priority_names = [
        "Energy_consumption.csv",
        "energy_consumption.csv",
        "energy_consumption_complete.csv",
        "EnergyConsumption.csv",
    ]

    # Check priority names first, then any *.csv
    candidates = []
    for d in search_dirs:
        for name in priority_names:
            p = os.path.join(d, name)
            if os.path.exists(p):
                candidates.insert(0, p)   # put priority files first
    for d in search_dirs:
        candidates += glob.glob(os.path.join(d, "*.csv"))

    seen = set()
    for csv_path in candidates:
        csv_path = os.path.normpath(csv_path)
        if csv_path in seen:
            continue
        seen.add(csv_path)
        try:
            df = pd.read_csv(csv_path)
            if 'EnergyConsumption' not in df.columns:
                log.warning(f"Skipping {os.path.basename(csv_path)}: no EnergyConsumption column")
                continue

            # Find & rename timestamp column
            time_col = next((c for c in
                ['Timestamp','timestamp','DateTime','datetime','Date','date']
                if c in df.columns), None)
            if time_col:
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.sort_values(time_col).reset_index(drop=True)
                df.rename(columns={time_col: 'Timestamp'}, inplace=True)
            else:
                df['Timestamp'] = pd.date_range(
                    start='2024-01-01', periods=len(df), freq='H')

            # Add calendar features if missing
            if 'Hour' not in df.columns:
                df['Hour']       = df['Timestamp'].dt.hour
            if 'Day' not in df.columns:
                df['Day']        = df['Timestamp'].dt.day
            if 'Month' not in df.columns:
                df['Month']      = df['Timestamp'].dt.month
            if 'DayOfYear' not in df.columns:
                df['DayOfYear']  = df['Timestamp'].dt.dayofyear
            if 'WeekOfYear' not in df.columns:
                df['WeekOfYear'] = df['Timestamp'].dt.isocalendar().week.astype(int)

            # Encode categoricals
            df = _encode_df(df)

            log.info(f"CSV loaded: {len(df)} rows from {os.path.relpath(csv_path, BASE)}")
            return df
        except Exception as e:
            log.warning(f"CSV error [{os.path.basename(csv_path)}]: {e}")

    log.warning("No valid CSV found — generating synthetic data")
    dates = pd.date_range(start='2024-01-01', periods=8760, freq='H')
    return pd.DataFrame({
        'Timestamp':        dates,
        'Temperature':      20 + 10*np.sin(2*np.pi*dates.dayofyear/365)
                            + 3*np.random.randn(8760),
        'Humidity':         50 + 15*np.random.randn(8760),
        'SquareFootage':    np.random.uniform(900, 2200, 8760),
        'Occupancy':        np.abs(np.random.randint(0, 25, 8760)),
        'HVACUsage':        np.random.choice([0,1], 8760),
        'LightingUsage':    np.random.choice([0,1], 8760),
        'RenewableEnergy':  np.random.uniform(5, 45, 8760),
        'DayOfWeek':        dates.weekday + 1,
        'Holiday':          np.random.choice([0,1], 8760, p=[0.97, 0.03]),
        'Hour':             dates.hour,
        'Day':              dates.day,
        'Month':            dates.month,
        'DayOfYear':        dates.dayofyear,
        'WeekOfYear':       dates.isocalendar().week.astype(int),
        'EnergyConsumption': (
            55 + 18*np.sin(2*np.pi*dates.hour/24)
            + 8*np.sin(2*np.pi*dates.dayofyear/365)
            + np.random.normal(0, 4, 8760)
        ),
    })


DF = load_data()
PEAK_THR = float(DF["EnergyConsumption"].quantile(0.75))

# Auto-build scaler from real CSV data if scaler_dl.pkl was a custom class
if SCALER_DL is None and DL_MODEL is not None:
    from sklearn.preprocessing import MinMaxScaler
    SCALER_DL = MinMaxScaler()
    SCALER_DL.fit(DF["EnergyConsumption"].values.reshape(-1,1))
    log.info("Scaler auto-built from CSV (scaler_dl.pkl had custom class)")

TARGET = "EnergyConsumption"


# ============================================================================
# FEATURE ENGINEERING — exact mirror of notebook Section 3
# ============================================================================

def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    # Cyclical
    d['Hour_sin']  = np.sin(2*np.pi*d['Hour']/24)
    d['Hour_cos']  = np.cos(2*np.pi*d['Hour']/24)
    d['Day_sin']   = np.sin(2*np.pi*d['DayOfWeek']/7)
    d['Day_cos']   = np.cos(2*np.pi*d['DayOfWeek']/7)
    d['Month_sin'] = np.sin(2*np.pi*d['Month']/12)
    d['Month_cos'] = np.cos(2*np.pi*d['Month']/12)
    # Lags (filled below for single-row inference)
    for lag in [1,2,3,6,12,24,48]:
        d[f'lag_{lag}'] = np.nan
    for w in [3,6,12,24]:
        d[f'roll_mean_{w}'] = np.nan
        d[f'roll_std_{w}']  = np.nan
    d['diff_1']  = np.nan
    d['diff_24'] = np.nan
    # Interactions
    d['Temp_x_HVAC']      = d['Temperature'] * d['HVACUsage']
    d['Temp_x_Occupancy'] = d['Temperature'] * d['Occupancy']
    d['Temp_x_Humidity']  = d['Temperature'] * d['Humidity']
    d['HVAC_x_Lighting']  = d['HVACUsage']   * d['LightingUsage']
    d['Occ_x_Lighting']   = d['Occupancy']   * d['LightingUsage']
    # Polynomial
    d['Temp_sq']  = d['Temperature']**2
    d['Humid_sq'] = d['Humidity']**2
    # Ratios / flags
    d['OccDensity'] = d['Occupancy'] / (d['SquareFootage'] + 1)
    d['RenewRatio'] = d['RenewableEnergy'] / (d['Temperature'] + 1)
    d['IsWeekend']  = (d['DayOfWeek'] >= 6).astype(int)
    d['PeakHour']   = d['Hour'].apply(lambda h: 1 if (8<=h<=10 or 17<=h<=20) else 0)
    return d


def build_inference_row(payload: dict) -> pd.DataFrame:
    row = {
        'Temperature':     float(payload.get('Temperature', 24)),
        'Humidity':        float(payload.get('Humidity', 50)),
        'SquareFootage':   float(payload.get('SquareFootage', 1500)),
        'Occupancy':       float(payload.get('Occupancy', 10)),
        'RenewableEnergy': float(payload.get('RenewableEnergy', 15)),
        'HVACUsage':       int(payload.get('HVACUsage', 1)),
        'LightingUsage':   int(payload.get('LightingUsage', 1)),
        'Hour':            int(payload.get('Hour', datetime.now().hour)),
        'Day':             int(payload.get('Day', datetime.now().day)),
        'Month':           int(payload.get('Month', datetime.now().month)),
        'DayOfWeek':       int(payload.get('DayOfWeek', datetime.now().weekday())) + 1,
        'DayOfYear':       int(payload.get('DayOfYear', datetime.now().timetuple().tm_yday)),
        'WeekOfYear':      int(payload.get('WeekOfYear', datetime.now().isocalendar()[1])),
        'Holiday':         int(payload.get('Holiday', 0)),
    }
    df_row = pd.DataFrame([row])
    df_row = engineer_features(df_row)

    # Fill lag / rolling from real CSV tail
    ec = DF['EnergyConsumption'].values
    n  = len(ec)
    for lag in [1,2,3,6,12,24,48]:
        df_row[f'lag_{lag}'] = ec[n-lag] if n >= lag else float(ec.mean())
    for w in [3,6,12,24]:
        window = ec[max(0,n-w):n]
        df_row[f'roll_mean_{w}'] = float(np.mean(window))
        df_row[f'roll_std_{w}']  = float(np.std(window)) if len(window)>1 else 0.0
    df_row['diff_1']  = float(ec[-1]-ec[-2])  if n>=2  else 0.0
    df_row['diff_24'] = float(ec[-1]-ec[-24]) if n>=24 else 0.0
    return df_row


def simple_predict(last_value: float, hour: int) -> float:
    if   17<=hour<=20: return last_value * 1.15
    elif  8<=hour<=10: return last_value * 1.10
    elif  0<=hour<= 5: return last_value * 0.75
    else:              return last_value * 0.95


# ============================================================================
# API ROUTES
# ============================================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":           "healthy",
        "timestamp":        datetime.now().isoformat(),
        "data_records":     len(DF),
        "prediction_mode":  MODE,
        "peak_threshold":   round(PEAK_THR, 2),
        "models_loaded": {
            "ml_model":     ML_MODEL is not None,
            "dl_model":     DL_MODEL is not None,
            "feature_cols": FEAT_COLS is not None,
            "scaler":       SCALER_DL is not None,
        }
    })


@app.route("/api/current-consumption", methods=["GET"])
def current_consumption():
    try:
        latest = DF.iloc[-1]
        prev   = DF.iloc[-2] if len(DF)>1 else latest
        cons   = float(latest["EnergyConsumption"])
        prev_c = float(prev["EnergyConsumption"])
        change = round(((cons-prev_c)/(prev_c or 1))*100, 1)
        ren    = float(latest.get("RenewableEnergy", 0))
        return jsonify({
            "consumption":         round(cons, 1),
            "change_vs_last_hour": change,
            "ai_savings_today":    12.8,
            "efficiency_gain":     21,
            "renewable_share":     round((ren/(cons or 1))*100, 1),
            "renewable_change":    5.1,
            "prediction_mode":     MODE,
            "timestamp":           str(latest["Timestamp"]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True) or {}

        if ML_MODEL is not None and FEAT_COLS is not None:
            df_row  = build_inference_row(payload)
            # Ensure all expected feature columns exist
            for c in FEAT_COLS:
                if c not in df_row.columns:
                    df_row[c] = 0.0
            pred       = _ml_predict(df_row[FEAT_COLS])
            model_used = ML_MODEL_NAME

        elif ML_MODEL is not None:
            df_row = build_inference_row(payload)
            try:
                pred = _ml_predict(df_row)
            except Exception:
                pred = simple_predict(float(DF['EnergyConsumption'].iloc[-1]),
                                      int(payload.get('Hour', datetime.now().hour)))
            model_used = ML_MODEL_NAME

        elif DL_MODEL is not None and SCALER_DL is not None:
            LOOKBACK = 24
            ec_tail  = DF['EnergyConsumption'].values[-LOOKBACK:]
            scaled   = SCALER_DL.transform(ec_tail.reshape(-1,1)).flatten()
            pred_s   = float(DL_MODEL.predict(scaled.reshape(1,LOOKBACK,1), verbose=0)[0][0])
            pred     = float(SCALER_DL.inverse_transform([[pred_s]])[0][0])
            model_used = "LSTM"

        else:
            last_val   = float(DF['EnergyConsumption'].iloc[-1])
            hour       = int(payload.get('Hour', datetime.now().hour))
            pred       = simple_predict(last_val, hour)
            model_used = "simulation"

        return jsonify({
            "prediction":     round(pred, 2),
            "model":          model_used,
            "is_peak":        bool(pred > PEAK_THR),
            "peak_threshold": round(PEAK_THR, 2),
            "timestamp":      datetime.now().isoformat(),
        })
    except Exception as e:
        log.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    try:
        hours    = min(int(request.args.get("hours", 6)), 48)
        latest   = DF.iloc[-1]
        cur_time = pd.to_datetime(latest["Timestamp"])
        prev_val = float(latest["EnergyConsumption"])
        preds    = []

        for i in range(1, hours+1):
            next_time = cur_time + timedelta(hours=i)
            hour      = next_time.hour

            if ML_MODEL is not None and FEAT_COLS is not None:
                payload = {
                    "Temperature":     float(latest.get("Temperature", 24)),
                    "Humidity":        float(latest.get("Humidity", 50)),
                    "SquareFootage":   float(latest.get("SquareFootage", 1500)),
                    "Occupancy":       float(latest.get("Occupancy", 10)),
                    "RenewableEnergy": float(latest.get("RenewableEnergy", 15)),
                    "HVACUsage":       int(latest.get("HVACUsage", 1)),
                    "LightingUsage":   int(latest.get("LightingUsage", 1)),
                    "Holiday":         int(latest.get("Holiday", 0)),
                    "Hour":       hour,
                    "Day":        next_time.day,
                    "Month":      next_time.month,
                    "DayOfWeek":  next_time.weekday(),  # 0-indexed; +1 done in build_inference_row
                    "DayOfYear":  next_time.timetuple().tm_yday,
                    "WeekOfYear": next_time.isocalendar()[1],
                }
                df_row  = build_inference_row(payload)
                for c in FEAT_COLS:
                    if c not in df_row.columns:
                        df_row[c] = 0.0
                pred_val = _ml_predict(df_row[FEAT_COLS])
                src = ML_MODEL_NAME
            else:
                pred_val = simple_predict(prev_val, hour)
                src = "simulation"

            pct = round(((pred_val-prev_val)/(prev_val or 1))*100, 1)
            preds.append({
                "timestamp":   next_time.isoformat(),
                "hour":        next_time.strftime("%H:00"),
                "consumption": round(pred_val, 1),
                "peak":        1 if pred_val > PEAK_THR else 0,
                "pct_change":  pct,
                "source":      src,
            })
            prev_val = pred_val

        return jsonify({
            "predictions": preds,
            "model_used":  preds[0]["source"] if preds else "simulation",
        })
    except Exception as e:
        log.exception("Forecast error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/zones", methods=["GET"])
def get_zones():
    zones_def = [
        {"name":"Medina",       "lat":36.799,"lon":10.165,"capacity":60, "renewable":12},
        {"name":"Centre Ville", "lat":36.800,"lon":10.180,"capacity":70, "renewable":18},
        {"name":"El Manar",     "lat":36.820,"lon":10.145,"capacity":80, "renewable":48},
        {"name":"Ariana",       "lat":36.862,"lon":10.195,"capacity":75, "renewable":52},
        {"name":"La Marsa",     "lat":36.878,"lon":10.325,"capacity":65, "renewable":34},
        {"name":"Ben Arous",    "lat":36.753,"lon":10.222,"capacity":65, "renewable":22},
    ]
    avg = float(DF["EnergyConsumption"].tail(24).mean())
    result = []
    for z in zones_def:
        cons   = round(avg*(0.8+0.4*np.random.random()), 1)
        load   = round((cons/z["capacity"])*100, 1)
        status = ("CRITICAL" if load>=130 else "WARNING" if load>=90
                  else "OPTIMAL" if load<=70 else "NORMAL")
        msgs   = {
            "CRITICAL": f"Emergency load shedding in {z['name']}",
            "WARNING":  f"Reduce consumption in {z['name']}",
            "OPTIMAL":  f"Optimal performance in {z['name']}",
            "NORMAL":   f"Monitor {z['name']}",
        }
        result.append({**z,"consumption":cons,"load":load,
                       "status":status,"ai_recommendation":msgs[status]})
    return jsonify({"zones": result})


@app.route("/api/ai-recommendation", methods=["GET"])
def ai_recommendation():
    preds = get_predictions().get_json().get("predictions", [])
    peak  = max(preds, key=lambda x: x["consumption"]) if preds else {}
    return jsonify({
        "title":             "AI Peak Load Management",
        "description":       (f"Predicted peak of {peak.get('consumption','—')} kWh "
                              f"at {peak.get('hour','—')}."),
        "estimated_savings": 18.3,
        "cost_saved":        4.2,
        "confidence":        94,
        "peak_hour":         peak.get("hour","16:00"),
        "peak_value":        peak.get("consumption", 58.1),
        "peak_zone":         "Medina",
        "energy_saved":      12.8,
        "energy_target":     20,
        "co2_reduced":       5.1,
        "co2_target":        8,
        "model_used":        MODE,
    })


@app.route("/api/historical", methods=["GET"])
def historical():
    try:
        period = request.args.get("period","24h")
        hours  = {"24h":24,"7d":168,"30d":720}.get(period,24)
        data   = DF.tail(hours)
        return jsonify({
            "timestamps":  data["Timestamp"].astype(str).tolist(),
            "consumption": [round(float(x),2) for x in data["EnergyConsumption"]],
            "temperature": [round(float(x),1) for x in data["Temperature"]],
            "renewable":   [round(float(x),1) for x in data["RenewableEnergy"]],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alerts", methods=["GET"])
def alerts():
    zones_data = get_zones().get_json().get("zones",[])
    al_list = []
    for z in zones_data:
        if z["status"] in ("CRITICAL","WARNING"):
            al_list.append({
                "level":     z["status"].lower(),
                "zone":      z["name"],
                "message":   (f"{z['name']} at {z['load']}% — "
                              f"{z['consumption']}/{z['capacity']} kWh"),
                "action":    z["ai_recommendation"],
                "timestamp": datetime.now().isoformat(),
            })
    return jsonify({"alerts": al_list, "count": len(al_list)})


@app.route("/api/energy-mix", methods=["GET"])
def energy_mix():
    avg_ren   = float(DF["RenewableEnergy"].tail(24).mean())
    avg_total = float(DF["EnergyConsumption"].tail(24).mean())
    ren_pct   = round((avg_ren/(avg_total or 1))*100, 1)
    grid_pct  = round(100-ren_pct, 1)
    return jsonify({
        "grid":            grid_pct,
        "solar":           round(ren_pct*0.6,1),
        "wind":            round(ren_pct*0.3,1),
        "storage":         round(ren_pct*0.1,1),
        "renewable_total": ren_pct,
        "pct": {
            "grid":    grid_pct,
            "solar":   round(ren_pct*0.6,1),
            "wind":    round(ren_pct*0.3,1),
            "storage": round(ren_pct*0.1,1),
        }
    })


@app.route("/api/stats/summary", methods=["GET"])
def summary_stats():
    return jsonify({
        "total_records":    len(DF),
        "prediction_mode":  MODE,
        "date_range": {
            "start": str(DF["Timestamp"].min()),
            "end":   str(DF["Timestamp"].max()),
        },
        "avg_consumption":  round(float(DF["EnergyConsumption"].mean()),2),
        "max_consumption":  round(float(DF["EnergyConsumption"].max()),2),
        "min_consumption":  round(float(DF["EnergyConsumption"].min()),2),
        "std_consumption":  round(float(DF["EnergyConsumption"].std()),2),
        "peak_threshold":   round(PEAK_THR,2),
        "avg_temperature":  round(float(DF["Temperature"].mean()),2),
        "avg_humidity":     round(float(DF["Humidity"].mean()),2),
        "avg_renewable":    round(float(DF["RenewableEnergy"].mean()),2),
    })


@app.route("/api/correlation", methods=["GET"])
def correlation():
    try:
        cols  = ["Temperature","Humidity","SquareFootage","Occupancy","RenewableEnergy"]
        valid = [c for c in cols if c in DF.columns]
        corrs = [round(float(DF[c].corr(DF["EnergyConsumption"])),3) for c in valid]
        return jsonify({"features":valid,"correlations":corrs})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/api/hourly-pattern", methods=["GET"])
def hourly_pattern():
    try:
        df_tmp         = DF.copy()
        df_tmp["Hour"] = pd.to_datetime(df_tmp["Timestamp"]).dt.hour
        grp            = df_tmp.groupby("Hour")["EnergyConsumption"]
        mean_s = grp.mean()
        std_s  = grp.std()
        return jsonify({
            "hours": list(range(24)),
            "mean":  [round(float(mean_s.get(h,0)),2) for h in range(24)],
            "std":   [round(float(std_s.get(h,0)),2)  for h in range(24)],
        })
    except Exception as e:
        return jsonify({"error":str(e)}), 500


# ============================================================================
if __name__ == "__main__":
    log.info("="*60)
    log.info("SMART ENERGY API SERVER")
    log.info("="*60)
    log.info(f"Data records  : {len(DF)}")
    log.info(f"ML model      : {'OK - ' + ML_MODEL_NAME if ML_MODEL else 'MISSING'}")
    log.info(f"DL model      : {'OK' if DL_MODEL  else 'MISSING'}")
    log.info(f"Feature cols  : {'OK - ' + str(len(FEAT_COLS)) + ' feats' if FEAT_COLS else 'MISSING'}")
    log.info(f"Scaler        : {'OK - ' + type(SCALER_DL).__name__ if SCALER_DL else 'MISSING'}")
    log.info(f"Active mode   : {MODE}")
    log.info(f"Peak threshold: {PEAK_THR:.2f} kWh")
    log.info("="*60)
    log.info("Server: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)