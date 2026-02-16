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
    """Calculates Heat Index or Wind Chill."""
    feels_like = T_f.copy()
    
    # Wind Chill (T < 50F, V > 3mph)
    mask_chill = (T_f < 50) & (V_mph > 3)
    feels_like = np.where(mask_chill,
        35.74 + 0.6215*T_f - 35.75*(V_mph**0.16) + 0.4275*T_f*(V_mph**0.16),
        feels_like
    )
    return feels_like

def create_plot(model, var_key, data, lons, lats, init_time, valid_time, fxx, cmap, vmin, vmax, label):
    try:
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

        filename = f"{model}_{var_key}_f{fxx:02d}.png"
        out_path = f"site/images/{filename}"
        plt.savefig(out_path, bbox_inches='tight', dpi=70) # Low DPI for speed
        plt.close()
        return f"images/{filename}"
    except Exception as e:
        print(f"      ! Plotting error ({var_key}): {e}")
        plt.close()
        return None

# --- MAIN LOOP ---
def process_model(name, config):
    print(f"\n--- Processing {name.upper()} ---")
    base_time = get_run_time(name)
    
    # 1. Find the latest available run
    found_run = False
    valid_dt = None
    
    for cycle_offset in range(4):
        step = 1 if name == 'hrrr' else 6
        run_dt = base_time - timedelta(hours=cycle_offset * step)
        try:
            print(f"  Checking {run_dt.strftime('%H')}z...")
            # Simple probe to check if run exists
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
        "frames": [] 
    }

    # 2. Loop through Forecast Hours
    for fxx in range(0, MAX_HOURS + 1):
        print(f"    Processing Forecast Hour: {fxx}")
        frame_data = {"fxx": fxx, "valid": (valid_dt + timedelta(hours=fxx)).strftime('%H:%M')}
        
        H = Herbie(valid_dt, model=config['model'], product=config['product'], fxx=fxx, verbose=False)

        # --- A. COMBINED DOWNLOAD: WIND + TEMP (The Fix) ---
        try:
            # We search for ALL variables we need for the complex calc at once.
            # TMP at 2m, UGRD/VGRD at 10m
            search_str = ":(TMP:2 m|UGRD:10 m|VGRD:10 m):"
            
            # This returns either a single Dataset or a list of Datasets (if levels differ)
            ds_mix = H.xarray(search_str, verbose=False)
            
            # Ensure we have a list to merge
            if not isinstance(ds_mix, list):
                ds_mix = [ds_mix]
            
            # Merge logic: 'override' forces 2m and 10m grids to align even if coords differ slightly
            ds_merged = xr.merge(ds_mix, compat='override')

            # Extract variables safely by standard name patterns
            # GRIB names: t2m (temp), u10/v10 (wind) OR t/u/v depending on model
            t_var = next((v for v in ds_merged.data_vars if 't' in v), None)
            u_var = next((v for v in ds_merged.data_vars if 'u' in v), None)
            v_var = next((v for v in ds_merged.data_vars if 'v' in v), None)

            if t_var and u_var and v_var:
                t_val = ds_merged[t_var]
                u_val = ds_merged[u_var]
                v_val = ds_merged[v_var]

                # Calculations
                ws_mph = np.sqrt(u_val**2 + v_val**2) * 2.237
                t_f = (t_val - 273.15) * 9/5 + 32
                apparent_t = calculate_apparent_temp(t_f.values, ws_mph.values)

                # Plot Wind
                p1 = create_plot(name, "wind", ws_mph, ds_merged.longitude, ds_merged.latitude, valid_dt, 
                               valid_dt + timedelta(hours=fxx), fxx, 'ocean_r', 0, 40, 'Wind Speed (mph)')
                frame_data['wind'] = p1
                
                # Plot Feels Like
                p2 = create_plot(name, "feelslike", apparent_t, ds_merged.longitude, ds_merged.latitude, valid_dt,
                               valid_dt + timedelta(hours=fxx), fxx, 'turbo', -20, 110, 'Apparent Temp (°F)')
                frame_data['feelslike'] = p2

        except Exception as e:
            # If wind fails, we just print the error and continue to other vars
            # It won't crash the whole script now.
            print(f"      ! Wind/Chill Skipped: {e}")

        # --- B. STANDARD VARIABLES ---
        for var_key, recipe in variables.items():
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
