import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from herbie import Herbie
from datetime import datetime, timedelta
import warnings
import os
import json

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# MODEL DEFINITIONS
# Each entry specifies:
#   model       - Herbie model ID
#   product     - GRIB2 product / stream
#   step        - hours between forecast frames
#   max_hours   - maximum forecast hour to attempt
#   cycle_hrs   - how often the model initialises (1 = hourly, 6 = 6-hourly, 12 = 12-hourly)
#   priority    - ordered list of data sources Herbie tries
#   member      - ensemble member (GEFS / ensemble models)
#   skip_vars   - variable keys to skip for this model (e.g. unavailable fields)
#   display_name- human-readable name shown in the viewer
# ---------------------------------------------------------------------------
MODELS = {
    # ── NOAA Deterministic ──────────────────────────────────────────────────
    'hrrr': {
        'model': 'hrrr', 'product': 'sfc',
        'step': 1, 'max_hours': 24, 'cycle_hrs': 1,
        'priority': ['aws', 'nomads'],
        'display_name': 'HRRR',
    },
    'rap': {
        'model': 'rap', 'product': 'awp130pgrb',
        'step': 1, 'max_hours': 21, 'cycle_hrs': 1,
        'priority': ['nomads'],
        'display_name': 'RAP',
    },
    'gfs': {
        'model': 'gfs', 'product': 'pgrb2.0p25',
        'step': 6, 'max_hours': 120, 'cycle_hrs': 6,
        'priority': ['aws', 'nomads'],
        'display_name': 'GFS',
    },
    'nam': {
        'model': 'nam', 'product': 'awip12',
        'step': 6, 'max_hours': 60, 'cycle_hrs': 6,
        'priority': ['nomads'],
        'display_name': 'NAM',
    },
    'nam3k': {
        'model': 'nam', 'product': 'conusnest.hiresf',
        'step': 6, 'max_hours': 60, 'cycle_hrs': 6,
        'priority': ['nomads'],
        'display_name': 'NAM 3km',
    },

    # ── NOAA Ensemble ────────────────────────────────────────────────────────
    'gefs': {
        'model': 'gefs', 'product': 'atmos.5',
        'step': 6, 'max_hours': 120, 'cycle_hrs': 6,
        'priority': ['aws'],
        'member': 'c00',
        'display_name': 'GEFS',
    },

    # ── NOAA AI / Experimental ───────────────────────────────────────────────
    # Project EAGLE – NOAA's AI global forecast system (aigfs)
    'aigfs': {
        'model': 'aigfs', 'product': 'pgrb2.0p25',
        'step': 6, 'max_hours': 120, 'cycle_hrs': 6,
        'priority': ['nomads'],
        'display_name': 'AIGFS (EAGLE)',
    },
    # AI GEFS – AI-enhanced ensemble control member
    'ai_gefs': {
        'model': 'gefs', 'product': 'atmos.25',
        'step': 6, 'max_hours': 120, 'cycle_hrs': 6,
        'priority': ['aws'],
        'member': 'c00',
        'display_name': 'AI GEFS',
    },

    # ── ECMWF Open Data ──────────────────────────────────────────────────────
    # IFS deterministic (HRES)
    'ecmwf': {
        'model': 'ifs', 'product': 'oper',
        'step': 6, 'max_hours': 120, 'cycle_hrs': 12,
        'priority': ['ecmwf'],
        'skip_vars': ['radar', 'snow'],
        'display_name': 'ECMWF IFS',
    },
    # AIFS – ECMWF's AI-based deterministic forecast
    'ecmwf_aifs': {
        'model': 'aifs', 'product': 'oper',
        'step': 6, 'max_hours': 120, 'cycle_hrs': 12,
        'priority': ['ecmwf'],
        'skip_vars': ['radar', 'gust', 'snow'],
        'display_name': 'ECMWF AIFS',
    },
    # IFS Ensemble (ENS)
    'ecmwf_ens': {
        'model': 'ifs', 'product': 'enfo',
        'step': 6, 'max_hours': 120, 'cycle_hrs': 12,
        'priority': ['ecmwf'],
        'skip_vars': ['radar', 'snow'],
        'display_name': 'ECMWF ENS',
    },
    # AIFS Ensemble
    'ecmwf_ai_ens': {
        'model': 'aifs', 'product': 'enfo',
        'step': 6, 'max_hours': 120, 'cycle_hrs': 12,
        'priority': ['ecmwf'],
        'skip_vars': ['radar', 'gust', 'snow'],
        'display_name': 'ECMWF AI ENS',
    },
}

# ---------------------------------------------------------------------------
# VARIABLE RECIPES
# ---------------------------------------------------------------------------
VARIABLES = {
    'temp': {
        'name': 'Temperature (2m)',
        'search': ':TMP:2 m',
        'cmap': 'turbo', 'vmin': -10, 'vmax': 100,
        'unit': 'F', 'accum': False,
    },
    'gust': {
        'name': 'Wind Gusts (sfc)',
        'search': ':GUST:surface',
        'cmap': 'Reds', 'vmin': 0, 'vmax': 60,
        'unit': 'mph', 'accum': False,
    },
    'radar': {
        'name': 'Simulated Radar',
        'search': ':REFC:',
        'cmap': 'gist_ncar', 'vmin': 0, 'vmax': 70,
        'unit': 'dBZ', 'accum': False,
    },
    'precip': {
        'name': '1-hr Total Precip',
        'search': ':APCP:',
        'cmap': 'YlGnBu', 'vmin': 0, 'vmax': 1.0,
        'unit': 'in', 'accum': True,
    },
    'snow': {
        'name': '1-hr Snowfall',
        'search': ':(ASNOW|WEASD):',
        'cmap': 'Blues', 'vmin': 0, 'vmax': 2.0,
        'unit': 'in', 'accum': True,
    },
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def get_run_time(config):
    """Return the expected latest model initialisation time (UTC)."""
    now = datetime.utcnow()
    cycle_hrs = config.get('cycle_hrs', 6)
    if cycle_hrs == 1:
        return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    else:
        hour = (now.hour // cycle_hrs) * cycle_hrs
        return now.replace(hour=hour, minute=0, second=0, microsecond=0)


def make_herbie(config, run_dt, fxx):
    """Build a Herbie object from a model config dict."""
    kwargs = dict(
        model=config['model'],
        fxx=fxx,
        verbose=False,
    )
    if config.get('product'):
        kwargs['product'] = config['product']
    if config.get('priority'):
        kwargs['priority'] = config['priority']
    if config.get('member') is not None:
        kwargs['member'] = config['member']
    return Herbie(run_dt, **kwargs)


def probe_available(config, run_dt, fxx=0):
    """Return True if data exists for run_dt / fxx, False otherwise."""
    try:
        H = make_herbie(config, run_dt, fxx)
        return H.grib is not None
    except Exception:
        return False


def calculate_apparent_temp(T_f, V_mph):
    """Wind-chill formula for T < 50 °F and V > 3 mph."""
    feels_like = T_f.copy()
    mask = (T_f < 50) & (V_mph > 3)
    feels_like = np.where(
        mask,
        35.74 + 0.6215 * T_f - 35.75 * (V_mph ** 0.16) + 0.4275 * T_f * (V_mph ** 0.16),
        feels_like,
    )
    return feels_like


def create_plot(model_key, var_key, data, lons, lats, init_time, valid_time, fxx, cmap, vmin, vmax, label):
    """Render data to a PNG and return the relative path, or None on failure."""
    try:
        fig = plt.figure(figsize=(10, 8))
        proj = ccrs.LambertConformal(central_longitude=-75, central_latitude=42)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([-82, -66, 37, 47], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        mesh = ax.pcolormesh(
            lons, lats, data,
            transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=vmin, vmax=vmax, shading='auto',
        )
        plt.colorbar(mesh, orientation='vertical', shrink=0.7, label=label, pad=0.02)

        plt.title(f"{model_key.upper()} f{fxx:02d}", loc='left', fontweight='bold', fontsize=12)
        plt.title(f"Valid: {valid_time.strftime('%a %H:00')}Z", loc='right', fontsize=9)

        filename = f"{model_key}_{var_key}_f{fxx:02d}.png"
        out_path = f"site/images/{filename}"
        plt.savefig(out_path, bbox_inches='tight', dpi=70)
        plt.close()
        return f"images/{filename}"
    except Exception as e:
        print(f"      ! Plot error ({var_key} f{fxx:02d}): {e}")
        plt.close()
        return None


def process_frame(model_key, config, run_dt, fxx):
    """
    Fetch and render ALL variables for a single forecast hour.
    Returns a frame_data dict (always contains 'fxx' and 'valid').
    """
    skip_vars = config.get('skip_vars', [])
    valid_time = run_dt + timedelta(hours=fxx)
    frame_data = {
        'fxx': fxx,
        'valid': valid_time.strftime('%H:%M'),
    }

    try:
        H = make_herbie(config, run_dt, fxx)
    except Exception as e:
        print(f"      ! Herbie init failed for f{fxx:02d}: {e}")
        return frame_data

    # ── A. Combined wind + temperature ──────────────────────────────────────
    if 'wind' not in skip_vars:
        try:
            ds_mix = H.xarray(":(TMP:2 m|UGRD:10 m|VGRD:10 m):", verbose=False)
            if not isinstance(ds_mix, list):
                ds_mix = [ds_mix]
            ds = xr.merge(ds_mix, compat='override')

            t_var = next((v for v in ds.data_vars if 't' in v), None)
            u_var = next((v for v in ds.data_vars if 'u' in v), None)
            v_var = next((v for v in ds.data_vars if 'v' in v), None)

            if t_var and u_var and v_var:
                ws_mph = np.sqrt(ds[u_var] ** 2 + ds[v_var] ** 2) * 2.237
                t_f = (ds[t_var] - 273.15) * 9 / 5 + 32
                apparent = calculate_apparent_temp(t_f.values, ws_mph.values)

                p1 = create_plot(model_key, 'wind', ws_mph, ds.longitude, ds.latitude,
                                 run_dt, valid_time, fxx, 'ocean_r', 0, 40, 'Wind Speed (mph)')
                frame_data['wind'] = p1

                p2 = create_plot(model_key, 'feelslike', apparent, ds.longitude, ds.latitude,
                                 run_dt, valid_time, fxx, 'turbo', -20, 110, 'Apparent Temp (°F)')
                frame_data['feelslike'] = p2
        except Exception as e:
            print(f"      ! Wind/FeelsLike skipped: {e}")

    # ── B. Standard variable recipes ────────────────────────────────────────
    for var_key, recipe in VARIABLES.items():
        if var_key in skip_vars:
            continue
        if fxx == 0 and recipe['accum']:
            continue  # Accumulated fields are zero at f00

        try:
            ds = H.xarray(recipe['search'], verbose=False)
            if ds is None or len(ds) == 0:
                continue
            var_name = list(ds.data_vars)[0]
            data = ds[var_name]

            if recipe['unit'] == 'F':
                data = (data - 273.15) * 9 / 5 + 32
            elif recipe['unit'] == 'mph':
                data = data * 2.237
            elif recipe['unit'] == 'in':
                data = data * 0.0393701

            p = create_plot(model_key, var_key, data, ds.longitude, ds.latitude,
                            run_dt, valid_time, fxx,
                            recipe['cmap'], recipe['vmin'], recipe['vmax'], recipe['name'])
            frame_data[var_key] = p
        except Exception:
            continue

    return frame_data


# ---------------------------------------------------------------------------
# SMART FRAME DETECTION
# ---------------------------------------------------------------------------

def load_existing_status():
    """Load status.json from the previous run (returns {} on any failure)."""
    path = 'site/status.json'
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def process_model(name, config, existing_status):
    """
    Process one model with smart frame detection:
      - If a NEW model run is found  → clear old data and process all frames.
      - If the SAME run is current   → detect only newly available forecast hours
                                       and process ALL variables for each new fxx.
    """
    print(f"\n─── {config['display_name']} ({name}) ─────────────────────────")

    step = config.get('step', 6)
    max_hours = config.get('max_hours', 24)
    cycle_hrs = config.get('cycle_hrs', 6)

    base_time = get_run_time(config)

    # ── 1. Find the latest available model run ───────────────────────────────
    valid_dt = None
    for offset in range(4):
        run_dt = base_time - timedelta(hours=offset * (1 if cycle_hrs == 1 else cycle_hrs))
        print(f"  Probing {run_dt.strftime('%Y-%m-%d %H')}z f00 …", end=' ')
        if probe_available(config, run_dt, fxx=0):
            valid_dt = run_dt
            print('✓')
            break
        print('✗')

    if valid_dt is None:
        print(f'  ✗ No data found — skipping {name}')
        return None

    new_run = valid_dt.strftime('%Y-%m-%d %H') + 'z'
    print(f'  → Locked on run: {new_run}')

    # ── 2. Compare with existing status.json ────────────────────────────────
    existing = (existing_status.get(name) or {})
    existing_run = existing.get('run_time', '')
    is_new_run = (new_run != existing_run)

    all_expected_fxxs = list(range(0, max_hours + 1, step))

    if is_new_run:
        print(f'  ★ New model run detected — processing all {len(all_expected_fxxs)} frames.')
        model_output = {
            'run_time': new_run,
            'display_name': config['display_name'],
            'frames': [],
        }
        frames_to_process = all_expected_fxxs
        existing_frames_map = {}

    else:
        print(f'  ↺ Same run as before — scanning for new forecast hours …')
        existing_frames = list(existing.get('frames', []))
        model_output = {
            'run_time': new_run,
            'display_name': config['display_name'],
            'frames': existing_frames,
        }
        existing_frames_map = {f['fxx']: f for f in existing_frames}

        # Walk forward from the first missing fxx; stop at first unavailable hour
        frames_to_process = []
        for fxx in all_expected_fxxs:
            if fxx in existing_frames_map:
                continue  # Already processed this hour
            print(f'    Probing f{fxx:03d} …', end=' ')
            if probe_available(config, valid_dt, fxx=fxx):
                print('✓ — queued')
                frames_to_process.append(fxx)
            else:
                print('✗ — stopping scan')
                break  # Data posts sequentially; no point checking further

        if not frames_to_process:
            print('  ✓ No new frames available — nothing to do.')
            return model_output

    print(f'  Processing {len(frames_to_process)} forecast hour(s): {frames_to_process}')

    # ── 3. Generate images for each new forecast hour ────────────────────────
    for fxx in frames_to_process:
        print(f'    fxx={fxx:03d}h …')
        frame_data = process_frame(name, config, valid_dt, fxx)

        if fxx in existing_frames_map:
            # Replace in-place
            for i, f in enumerate(model_output['frames']):
                if f['fxx'] == fxx:
                    model_output['frames'][i] = frame_data
                    break
        else:
            model_output['frames'].append(frame_data)

    # Keep frames sorted
    model_output['frames'].sort(key=lambda x: x['fxx'])
    return model_output


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs('site/images', exist_ok=True)

    existing_status = load_existing_status()
    status_report = {}

    for model_name, config in MODELS.items():
        result = process_model(model_name, config, existing_status)
        status_report[model_name] = result

    with open('site/status.json', 'w') as f:
        json.dump(status_report, f, indent=2)

    print('\n✅  status.json updated.')
