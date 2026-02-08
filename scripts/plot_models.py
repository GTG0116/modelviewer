import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie
from datetime import datetime, timedelta
import warnings
import os
import json
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

def get_run_time(model_name):
    """Snaps to the latest likely available run."""
    now = datetime.utcnow()
    
    if model_name in ['hrrr', 'rrfs']:
        # HRRR runs hourly. Back up 1 hour.
        return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    else:
        # GFS/NAM run every 6 hours (00, 06, 12, 18).
        hour = (now.hour // 6) * 6
        return now.replace(hour=hour, minute=0, second=0, microsecond=0)

def plot_model(name, config):
    print(f"\n--- Processing {name.upper()} ---")
    base_time = get_run_time(name)
    
    # Check current cycle and previous ones if needed
    for cycle_offset in range(4):
        if name == 'hrrr':
            run_dt = base_time - timedelta(hours=cycle_offset)
        else:
            run_dt = base_time - timedelta(hours=cycle_offset * 6)
            
        try:
            print(f"  Attempting cycle: {run_dt.strftime('%Y-%m-%d %H:00')} UTC...")
            
            # 1. Initialize Herbie
            H = Herbie(
                run_dt,
                model=config['model'],
                product=config['product'],
                priority=['aws', 'nomads'], 
                save_dir='./data',
                fxx=0,
                verbose=False
            )

            # 2. Key Filtering
            # Crucial for GRIB2 files to select the "Instant" temperature
            backend_kwargs = {
                'filter_by_keys': {
                    'typeOfLevel': 'heightAboveGround',
                    'level': 2,
                    'stepType': 'instant' 
                }
            }

            # 3. Download & Open
            ds = H.xarray(
                config['search'], 
                backend_kwargs=backend_kwargs, 
                verbose=False
            )

            if ds is None or len(ds.data_vars) == 0:
                print("    ! Dataset empty.")
                continue
                
            # Grab the variable (t, t2m, etc.)
            var_name = list(ds.data_vars)[0]
            data = ds[var_name]

            # 4. Plotting
            data_f = (data - 273.15) * 9/5 + 32
            
            fig = plt.figure(figsize=(10, 8))
            
            # Lambert Conformal for Northeast US
            proj = ccrs.LambertConformal(central_longitude=-75, central_latitude=42)
            ax = fig.add_subplot(1, 1, 1, projection=proj)
            
            # [West, East, South, North]
            ax.set_extent([-82, -66, 37, 47], crs=ccrs.PlateCarree())

            # Features
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':')

            # Plot Data
            mesh = ax.pcolormesh(ds.longitude, ds.latitude, data_f,
                                 transform=ccrs.PlateCarree(),
                                 cmap='turbo', vmin=-10, vmax=100, shading='auto')
            
            plt.colorbar(mesh, orientation='vertical', shrink=0.7, label='Temperature (°F)')
            
            title_time = run_dt.strftime('%d %b %H:00Z')
            plt.title(f"{name.upper()} 2m Temp", loc='left', fontweight='bold')
            plt.title(f"Run: {title_time}", loc='right')

            # Output
            out_file = f"site/images/{name}.png"
            os.makedirs("site/images", exist_ok=True)
            plt.savefig(out_file, bbox_inches='tight', dpi=100)
            plt.close()
            
            print(f"  ✅ SUCCESS: Saved {out_file}")
            
            return {
                "status": "success",
                "run": f"{title_time}",
                "image": f"images/{name}.png"
            }

        except Exception as e:
            print(f"    ! Error: {e}")
            continue

    print(f"  ❌ All attempts failed for {name}")
    return {"status": "failed", "image": "images/placeholder.png"}

# --- UPDATED CONFIGURATION ---
models = {
    'hrrr': {
        'model': 'hrrr', 
        'product': 'sfc', 
        'search': ':TMP:2 m'
    },
    'gfs': {
        'model': 'gfs', 
        'product': 'pgrb2.0p25', 
        'search': ':TMP:2 m'
    },
    # CORRECTED NAM PRODUCT NAMES
    'nam': {
        'model': 'nam', 
        'product': 'awip12',  # Changed from 'conus'
        'search': ':TMP:2 m'
    },
    'nam3k': {
        'model': 'nam', 
        'product': 'conusnest.hiresf', # Changed from 'conusnest'
        'search': ':TMP:2 m'
    }
}

if __name__ == "__main__":
    os.makedirs("site/images", exist_ok=True)
    
    status_report = {}
    for model_name, config in models.items():
        status_report[model_name] = plot_model(model_name, config)
    
    with open("site/status.json", "w") as f:
        json.dump(status_report, f, indent=2)
