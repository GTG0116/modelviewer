import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SAVE_DIR = "herbie_data"
# Northeast US View
EXTENTS = [-80.5, -71.5, 38.5, 43.5] 

# Map Projection
MAP_CRS = ccrs.LambertConformal(central_longitude=-76.0, central_latitude=41.0)

# Variables to Plot
VARIABLES = {
    "temp":   {"search": ":TMP:2 m", "cmap": "turbo", "min": 0, "max": 100, "label": "Temperature (°F)"},
    "wind":   {"search": ":(UGRD|VGRD):10 m", "cmap": "viridis", "min": 0, "max": 50, "label": "Wind Speed (mph)"},
    "refc":   {"search": ":REFC:entire", "cmap": "pyart_NWSRef", "min": 0, "max": 70, "label": "Simulated Radar (dBZ)"},
    "snow":   {"search": ":(ASNOW|WEASD):surface", "cmap": "Blues", "min": 0, "max": 24, "label": "Snowfall (in)"}
}

# --- HELPER FUNCTIONS ---
def get_latest_run_time(model_type):
    """Finds the latest available run for a specific model."""
    now = datetime.utcnow()
    for i in range(12): # Look back up to 12 hours
        t = now - timedelta(hours=i)
        try:
            # Check if F01 exists to confirm the run is "ready"
            H = Herbie(t.strftime('%Y-%m-%d %H:00'), model=model_type, fxx=1, save_dir=SAVE_DIR)
            if H.inventory() is not None:
                return t
        except:
            continue
    return None

def process_model(model_name):
    run_time = get_latest_run_time(model_name)
    if not run_time:
        print(f"Could not find recent data for {model_name}")
        return

    # Determine max forecast length
    # RRFS/HRRR go long (48h/60h) at 00, 06, 12, 18z; otherwise 18h.
    run_hour = int(run_time.strftime("%H"))
    is_long_run = run_hour in [0, 6, 12, 18]
    
    if model_name == "hrrr":
        max_fxx = 48 if is_long_run else 18
    else: # rrfs
        max_fxx = 60 if is_long_run else 18

    print(f"Processing {model_name.upper()} | Run: {run_time.strftime('%H')}Z | Frames: {max_fxx}")

    for fxx in range(1, max_fxx + 1):
        try:
            H = Herbie(run_time, model=model_name, fxx=fxx, save_dir=SAVE_DIR)
            
            for var_name, cfg in VARIABLES.items():
                folder = f"frames_{model_name}_{var_name}"
                os.makedirs(folder, exist_ok=True)
                
                # Fetch Data
                ds = H.xarray(cfg["search"], verbose=False)
                
                # --- DATA PROCESSING ---
                # 1. Temperature conversion (K -> F)
                if var_name == "temp":
                    data = (ds.t2m - 273.15) * 9/5 + 32
                # 2. Wind Speed (m/s -> mph)
                elif var_name == "wind":
                    u = ds['u10'] if 'u10' in ds else ds['u']
                    v = ds['v10'] if 'v10' in ds else ds['v']
                    data = np.sqrt(u**2 + v**2) * 2.237
                # 3. Snow (m -> in)
                elif var_name == "snow":
                    # RRFS sometimes uses 'unknown' or different names
                    raw = ds[list(ds.data_vars)[0]] 
                    data = raw * 39.37
                # 4. Reflectivity
                elif var_name == "refc":
                    data = ds[list(ds.data_vars)[0]]

                # --- PLOTTING ---
                fig = plt.figure(figsize=(10, 8), facecolor='black')
                ax = plt.axes(projection=MAP_CRS)
                ax.set_extent(EXTENTS, crs=ccrs.PlateCarree())
                
                # Overlays
                ax.add_feature(cfeature.STATES.with_scale('10m'), linewidth=0.5, edgecolor='white')
                ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, edgecolor='white')

                # Plot Data
                # Gaussian smooth slightly for nicer look (except radar)
                plot_data = data if var_name == "refc" else gaussian_filter(data, sigma=1)
                
                im = ax.pcolormesh(ds.longitude, ds.latitude, plot_data, transform=ccrs.PlateCarree(),
                                   cmap=cfg['cmap'], vmin=cfg['min'], vmax=cfg['max'], shading='auto')
                
                # Annotation
                plt.title(f"{model_name.upper()} {cfg['label']} | F{fxx:02d}", color='white', loc='left')
                plt.title(f"Run: {run_time.strftime('%Y-%m-%d %H')}Z", color='gray', loc='right')

                # Save
                filename = f"{folder}/f{fxx:02d}.png"
                plt.savefig(filename, bbox_inches='tight', facecolor='black', dpi=90)
                plt.close()
                
                print(f"Saved {filename}")

        except Exception as e:
            print(f"Error on {model_name} F{fxx}: {e}")

if __name__ == "__main__":
    process_model("hrrr")
    process_model("rrfs")
