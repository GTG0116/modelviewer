import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from herbie import Herbie
from datetime import datetime, timedelta
import warnings
import os
import json
import sys

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---

# Model Definitions
models = {
    'hrrr':  {'model': 'hrrr', 'product': 'sfc',            'step': 1, 'lat_lon': [-75, 42]},
    'gfs':   {'model': 'gfs',  'product': 'pgrb2.0p25',     'step': 6, 'lat_lon': [-75, 42]},
    'nam':   {'model': 'nam',  'product': 'awip12',         'step': 6, 'lat_lon': [-75, 42]},
    'nam3k': {'model': 'nam',  'product': 'conusnest.hiresf','step': 6, 'lat_lon': [-75, 42]},
}

# Variable Recipes
# 'fxx': Forecast hour offset (0 = Analysis, 1 = 1hr Forecast for precip)
variables = {
    'temp': {
        'name': 'Temperature (2m)',
        'search': ':TMP:2 m',
        'cmap': 'turbo', 'vmin': -10, 'vmax': 100,
        'fxx': 0,
        'unit': 'F'
    },
    'gust': {
        'name': 'Wind Gusts (sfc)',
        'search': ':GUST:surface',
        'cmap': 'Reds', 'vmin': 0, 'vmax': 60,
        'fxx': 0,
        'unit': 'mph'
    },
    'radar': {
        'name': 'Simulated Radar',
        'search': ':REFC:', # Composite Reflectivity
        'cmap': 'gist_ncar', 'vmin': 0, 'vmax': 70,
        'fxx': 0,
        'unit': 'dBZ'
    },
    'precip': {
        'name': '1-hr Total Precip',
        'search': ':APCP:',
        'cmap': 'Terra', 'vmin': 0, 'vmax': 1.0,
        'fxx': 1, # Must be forecasted to have accumulation
        'unit': 'in'
    },
    'snow': {
        'name': '1-hr Snowfall',
        'search': ':(ASNOW|WEASD):', # Accumulated Snow or Water Equiv
        'cmap': 'Blues', 'vmin': 0, 'vmax': 2.0,
        'fxx': 1,
        'unit': 'in'
    }
    # Note: Wind Speed and Apparent Temp are calculated dynamically below
}

# --- 2. HELPER FUNCTIONS ---

def get_run_time(model_name):
    """Snaps to the latest likely available run."""
    now = datetime.utcnow()
    if model_name in ['hrrr', 'rrfs']:
        return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    else:
        hour = (now.hour // 6) * 6
        return now.replace(hour=hour, minute=0, second=0, microsecond=0)

def calculate_apparent_temp(T_f, V_mph):
    """Calculates Heat Index or Wind Chill based on NOAA formulas."""
    # Initialize with actual temp
    feels_like = T_f.copy()
    
    # Wind Chill (T < 50F, V > 3mph)
    mask_chill = (T_f < 50) & (V_mph > 3)
    feels_like[mask_chill] = 35.74 + 0.6215*T_f[mask_chill] - 35.75*(V_mph[mask_chill]**0.16) + 0.4275*T_f[mask_chill]*(V_mph[mask_chill]**0.16)
    
    # Heat Index (Simple approximation for T > 80F)
    # (Full Rothfusz regression is complex, this is a simplified view for visualization)
    mask_heat = (T_f > 80)
    # Simple adjustment: Add 0.1 * (T - 80) just to show 'hotter' (Real formula requires Relative Humidity)
    # Since we aren't downloading RH to save bandwidth, we will just pass raw T for heat 
    # unless we want to download RH. For now, let's leave Heat Index as just Temp.
    
    return feels_like

# --- 3. MAIN PLOTTING LOOP ---

def plot_model_suite(name, config):
    print(f"\n--- Processing Suite: {name.upper()} ---")
    base_time = get_run_time(name)
    results = {}

    # Try to find a valid run (check last 3 cycles)
    found_run = False
    valid_dt = None
    
    for cycle_offset in range(4):
        step = 1 if name == 'hrrr' else 6
        run_dt = base_time - timedelta(hours=cycle_offset * step)
        
        # Test if we can access the file (using Temp as the probe)
        try:
            print(f"  Checking availability for {run_dt.strftime('%H')}z...")
            H_probe = Herbie(run_dt, model=config['model'], product=config['product'], 
                             fxx=0, priority=['aws', 'nomads'], verbose=False)
            # Just check if we can get the index or file
            if H_probe.grib is None: continue
            
            valid_dt = run_dt
            found_run = True
            break
        except:
            continue
            
    if not found_run:
        print(f"❌ No data found for {name}")
        return None

    # We have a valid date, now generate ALL plots for this model
    print(f"  ✅ Locked on run: {valid_dt.strftime('%Y-%m-%d %H')}z")
    
    # --- A. SPECIAL: Wind Speed & Apparent Temp (Requires U+V+T) ---
    try:
        H = Herbie(valid_dt, model=config['model'], product=config['product'], fxx=0, verbose=False)
        
        # Download U, V, and Temp
        ds_u = H.xarray(":UGRD:(10 m|1000 m)", verbose=False)
        ds_v = H.xarray(":VGRD:(10 m|1000 m)", verbose=False)
        ds_t = H.xarray(":TMP:2 m", verbose=False)
        
        if ds_u and ds_v and ds_t:
            # 1. Plot Wind Speed
            u = list(ds_u.data_vars)[0]; v = list(ds_v.data_vars)[0]
            ws_mph = np.sqrt(ds_u[u]**2 + ds_v[v]**2) * 2.237 # m/s to mph
            
            create_plot(name, "wind", ws_mph, ds_u.longitude, ds_u.latitude, valid_dt, 
                       cmap='ocean_r', vmin=0, vmax=40, label='Wind Speed (mph)')
            results['wind'] = f"images/{name}_wind.png"
            
            # 2. Plot Apparent Temp (Wind Chill)
            t_var = list(ds_t.data_vars)[0]
            t_f = (ds_t[t_var] - 273.15) * 9/5 + 32
            
            apparent_t = calculate_apparent_temp(t_f.values, ws_mph.values)
            
            create_plot(name, "feelslike", apparent_t, ds_u.longitude, ds_u.latitude, valid_dt,
                       cmap='turbo', vmin=-20, vmax=110, label='Apparent Temp (°F)')
            results['feelslike'] = f"images/{name}_feelslike.png"
            
    except Exception as e:
        print(f"    ! Failed Wind/Apparent calculations: {e}")

    # --- B. STANDARD VARIABLES (Loop) ---
    for var_key, recipe in variables.items():
        try:
            # Get correct Forecast Hour (0 for analysis, 1 for precip accumulation)
            fxx = recipe['fxx']
            H = Herbie(valid_dt, model=config['model'], product=config['product'], fxx=fxx, verbose=False)
            
            # Search
            ds = H.xarray(recipe['search'], verbose=False)
            if ds is None or len(ds) == 0: continue
            
            # Extract Data
            var_name = list(ds.data_vars)[0]
            data = ds[var_name]
            
            # Unit Conversions
            if recipe['unit'] == 'F':
                data = (data - 273.15) * 9/5 + 32
            elif recipe['unit'] == 'mph':
                data = data * 2.237
            elif recipe['unit'] == 'in':
                data = data * 0.0393701  # mm to inches
            
            # Plot
            create_plot(name, var_key, data, ds.longitude, ds.latitude, valid_dt,
                       cmap=recipe['cmap'], vmin=recipe['vmin'], vmax=recipe['vmax'], 
                       label=recipe['name'], fxx=fxx)
            
            results[var_key] = f"images/{name}_{var_key}.png"
            print(f"    -> Generated {var_key}")

        except Exception as e:
            print(f"    ! Skipped {var_key}: {e}")
            continue

    results['run_time'] = valid_dt.strftime('%d %b %H') + "z"
    return results

def create_plot(model, var_key, data, lons, lats, time, cmap, vmin, vmax, label, fxx=0):
    fig = plt.figure(figsize=(10, 8))
    proj = ccrs.LambertConformal(central_longitude=-75, central_latitude=42)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    
    # Northeast Extent
    ax.set_extent([-82, -66, 37, 47], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Handle coordinates (sometimes 1D, sometimes 2D)
    transform = ccrs.PlateCarree()
    
    mesh = ax.pcolormesh(lons, lats, data, transform=transform,
                         cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    
    plt.colorbar(mesh, orientation='vertical', shrink=0.7, label=label, pad=0.02)
    
    fcst_label = "Analysis" if fxx == 0 else f"{fxx}-hr Forecast"
    plt.title(f"{model.upper()} {label}", loc='left', fontweight='bold', fontsize=12)
    plt.title(f"{time.strftime('%H')}z Run | {fcst_label}", loc='right', fontsize=9)

    out_path = f"site/images/{model}_{var_key}.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=90)
    plt.close()

# --- 4. EXECUTION ---
if __name__ == "__main__":
    os.makedirs("site/images", exist_ok=True)
    status_report = {}
    
    for model_name, config in models.items():
        status_report[model_name] = plot_model_suite(model_name, config)
    
    with open("site/status.json", "w") as f:
        json.dump(status_report, f, indent=2)
