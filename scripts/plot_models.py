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
import sys

warnings.filterwarnings("ignore")

# --- GLOBAL SETTINGS ---
# Set this to 84 (NAM), 48 (HRRR), or 120+ (GFS) to get full runs.
# For now, we cap it at 24 to save GitHub Action minutes.
MAX_HOURS = 24 

# --- MODEL DEFINITIONS ---
models = {
    'hrrr':  {'model': 'hrrr', 'product': 'sfc',            'step': 1},
    'gfs':   {'model': 'gfs',  'product': 'pgrb2.0p25',     'step': 6},
    'nam':   {'model': 'nam',  'product': 'awip12',         'step': 6},
    'nam3k': {'model': 'nam',  'product': 'conusnest.hiresf','step': 6},
}

# --- VARIABLE RECIPES ---
variables = {
    'temp': {
        'name': 'Temperature (2m)',
        'search': ':TMP:2 m',
        'cmap': 'turbo', 'vmin': -10, 'vmax': 100,
        'unit': 'F',
        'accum': False
    },
    'gust': {
        'name': 'Wind Gusts (sfc)',
        'search': ':GUST:surface',
        'cmap': 'Reds', 'vmin': 0, 'vmax': 60,
        'unit': 'mph',
        'accum': False
    },
    'radar': {
        'name': 'Simulated Radar',
        'search': ':REFC:', # Composite Reflectivity
        'cmap': 'gist_ncar', 'vmin': 0, 'vmax': 70,
        'unit': 'dBZ',
        'accum': False
    },
    'precip': {
        'name': '1-hr Total Precip',
        'search': ':APCP:',
        'cmap': 'Terra', 'vmin': 0, 'vmax': 1.0,
        'unit': 'in',
        'accum': True
    },
    'snow': {
        'name': '1-hr Snowfall',
        'search': ':(ASNOW|WEASD):', 
        'cmap': 'Blues', 'vmin': 0, 'vmax': 2.0,
        'unit': 'in',
        'accum': True
    }
}

# --- HELPER FUNCTIONS ---
def get_run_time(model_name):
    now = datetime.utcnow()
    if model_name in ['hrrr']:
        return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    else:
        hour = (now.hour // 6) * 6
        return now.replace(hour=hour, minute=0, second=0, microsecond=0)

def calculate_apparent_temp(T_f, V_mph):
    """Calculates Heat Index or Wind Chill based on NOAA formulas."""
    feels_like = T_f.copy()
    
    # Wind Chill (T < 50F, V > 3mph)
    # We use numpy.where to avoid errors with NaNs or shapes
    mask_chill = (T_f < 50) & (V_mph > 3)
    
    feels_like = np.where(mask_chill,
        35.74 + 0.6215*T_f - 35.75*(V_mph**0.16) + 0.4275*T_f*(V_mph**0.16),
        feels_like
    )
    return feels_like

def create_plot(model, var_key, data, lons, lats, init_time, valid_time, fxx, cmap, vmin, vmax, label):
    fig = plt.figure(figsize=(10, 8))
    proj = ccrs.LambertConformal(central_longitude=-75, central_latitude=42)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-82, -66, 37, 47], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    mesh = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(),
                         cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    
    plt.colorbar(mesh, orientation='vertical', shrink=0.7, label=label, pad=0.02)
    
    valid_str = valid_time.strftime('%a %H:00')
    plt.title(f"{model.upper()} f{fxx:02d}", loc='left', fontweight='bold', fontsize=12)
    plt.title(f"Valid: {valid_str}Z", loc='right', fontsize=9)

    # Save as images/hrrr_temp_f01.png
    filename = f"{model}_{var_key}_f{fxx:02d}.png"
    out_path = f"site/images/{filename}"
    plt.savefig(out_path, bbox_inches='tight', dpi=80) # Lower DPI for speed
    plt.close()
    return f"images/{filename}"

# --- MAIN LOOP ---
def process_model(name, config):
    print(f"\n--- Processing {name.upper()} ---")
    base_time = get_run_time(name)
    
    # 1. Find the latest available run
    found_run = False
    valid_dt = None
    H_probe = None
    
    for cycle_offset in range(4):
        step = 1 if name == 'hrrr' else 6
        run_dt = base_time - timedelta(hours=cycle_offset * step)
        try:
            print(f"  Checking {run_dt.strftime('%H')}z...")
            H_probe = Herbie(run_dt, model=config['model'], product=config['product'], 
                             fxx=0, priority=['aws', 'nomads'], verbose=False)
            if H_probe.grib is None: continue
            valid_dt = run_dt
            found_run = True
            break
        except: continue
            
    if not found_run:
        print(f"❌ No data found for {name}")
        return None

    print(f"  ✅ Locked on run: {valid_dt.strftime('%Y-%m-%d %H')}z")
    
    model_output = {
        "run_time": valid_dt.strftime('%Y-%m-%d %H') + "z",
        "frames": [] # Will store { fxx: 0, temp: 'path', wind: 'path'... }
    }

    # 2. Loop through Forecast Hours
    # We step by 1 hour. For global models (GFS), you might want to step by 3 later.
    for fxx in range(0, MAX_HOURS + 1):
        print(f"    Processing Forecast Hour: {fxx}")
        
        frame_data = {"fxx": fxx, "valid": (valid_dt + timedelta(hours=fxx)).strftime('%H:%M')}
        H = Herbie(valid_dt, model=config['model'], product=config['product'], fxx=fxx, verbose=False)

        # --- A. SPECIAL: Wind Speed & Apparent Temp ---
        try:
            # Download components
            ds_u = H.xarray(":UGRD:(10 m|1000 m)", verbose=False)
            ds_v = H.xarray(":VGRD:(10 m|1000 m)", verbose=False)
            ds_t = H.xarray(":TMP:2 m", verbose=False)

            if ds_u and ds_v and ds_t:
                # ALIGN GRIDS (Crucial Fix for Wind Chill)
                # Sometimes U/V grids are slightly shifted from T grids. This forces them to match.
                ds_u, ds_v, ds_t = xr.align(ds_u, ds_v, ds_t, join='override')

                u = list(ds_u.data_vars)[0]
                v = list(ds_v.data_vars)[0]
                t = list(ds_t.data_vars)[0]
                
                # Math
                ws_mph = np.sqrt(ds_u[u]**2 + ds_v[v]**2) * 2.237
                t_f = (ds_t[t] - 273.15) * 9/5 + 32
                
                # Wind Chill Calculation
                apparent_t = calculate_apparent_temp(t_f.values, ws_mph.values)

                # Plot Wind
                p = create_plot(name, "wind", ws_mph, ds_u.longitude, ds_u.latitude, valid_dt, 
                               valid_dt + timedelta(hours=fxx), fxx, 'ocean_r', 0, 40, 'Wind Speed (mph)')
                frame_data['wind'] = p
                
                # Plot Feels Like
                p = create_plot(name, "feelslike", apparent_t, ds_u.longitude, ds_u.latitude, valid_dt,
                               valid_dt + timedelta(hours=fxx), fxx, 'turbo', -20, 110, 'Apparent Temp (°F)')
                frame_data['feelslike'] = p

        except Exception as e:
            print(f"      ! Wind/Chill failed: {e}")

        # --- B. STANDARD VARIABLES ---
        for var_key, recipe in variables.items():
            # Skip accum variables (precip/snow) at hour 0
            if fxx == 0 and recipe['accum']: continue
            
            try:
                ds = H.xarray(recipe['search'], verbose=False)
                if ds is None or len(ds) == 0: continue
                
                var_name = list(ds.data_vars)[0]
                data = ds[var_name]
                
                if recipe['unit'] == 'F': data = (data - 273.15) * 9/5 + 32
                elif recipe['unit'] == 'mph': data = data * 2.237
                elif recipe['unit'] == 'in': data = data * 0.0393701 

                p = create_plot(name, var_key, data, ds.longitude, ds.latitude, valid_dt,
                               valid_dt + timedelta(hours=fxx), fxx, recipe['cmap'], recipe['vmin'], recipe['vmax'], recipe['name'])
                frame_data[var_key] = p

            except: continue
        
        model_output['frames'].append(frame_data)

    return model_output

if __name__ == "__main__":
    os.makedirs("site/images", exist_ok=True)
    status_report = {}
    
    for model_name, config in models.items():
        status_report[model_name] = process_model(model_name, config)
    
    with open("site/status.json", "w") as f:
        json.dump(status_report, f, indent=2)
