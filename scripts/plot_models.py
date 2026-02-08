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
        # HRRR/RRFS: Runs hourly. Back up 2 hours to ensure upload is complete.
        return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    else:
        # GFS/NAM: Runs every 6 hours (00, 06, 12, 18).
        # We find the previous cycle floor.
        hour = (now.hour // 6) * 6
        return now.replace(hour=hour, minute=0, second=0, microsecond=0)

def plot_model(name, config):
    print(f"\n--- Processing {name.upper()} ---")
    base_time = get_run_time(name)
    
    # Try the last 3 cycles. If the current one isn't up, try the previous.
    for cycle_offset in range(3):
        # Calculate the run time to search for
        if name in ['hrrr', 'rrfs']:
            run_dt = base_time - timedelta(hours=cycle_offset)
        else:
            run_dt = base_time - timedelta(hours=cycle_offset * 6)
            
        try:
            print(f"  Attempting cycle: {run_dt.strftime('%Y-%m-%d %H:00')} UTC...")
            
            # 1. Initialize Herbie
            # We use 'aws' first. If it fails, Herbie might try others if configured,
            # but for GitHub Actions, we want to stick to AWS/NOMADS.
            H = Herbie(
                run_dt,
                model=config['model'],
                product=config['product'],
                priority=['aws', 'nomads'], 
                save_dir='./data',
                fxx=0, # Analysis hour
                verbose=False
            )

            # 2. Key Filtering (THE FIX FOR GFS/HRRR)
            # We must tell cfgrib EXACTLY which message to read to prevent crashes.
            # GRIBs have multiple "t" variables. We want 2m height.
            backend_kwargs = {
                'filter_by_keys': {
                    'typeOfLevel': 'heightAboveGround',
                    'level': 2,
                    'stepType': 'instant' # Ensures we don't get averages/accumulations
                }
            }

            # 3. Download & Open
            # strict=False allows it to fail gracefully if the index is bad
            try:
                ds = H.xarray(
                    config['search'], 
                    backend_kwargs=backend_kwargs, 
                    verbose=False
                )
            except Exception as read_err:
                print(f"    ! Download/Read failed: {read_err}")
                continue

            # 4. Check for Data
            if ds is None or len(ds.data_vars) == 0:
                print("    ! Dataset opened but is empty.")
                continue
                
            # Find the temperature variable (could be t, t2m, tp, etc.)
            # We look for the first variable in the dataset
            var_name = list(ds.data_vars)[0]
            print(f"    Found variable: {var_name}")
            data = ds[var_name]

            # 5. Plotting
            print("    Generating plot...")
            
            # Convert Kelvin to Fahrenheit
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
            print(f"    ! Error processing cycle: {e}")
            # import traceback
            # traceback.print_exc()
            continue

    print(f"  ❌ All attempts failed for {name}")
    return {"status": "failed", "image": "images/placeholder.png"}

# --- Configuration ---
models = {
    # HRRR: High Resolution Rapid Refresh
    'hrrr': {
        'model': 'hrrr', 
        'product': 'sfc', 
        'search': ':TMP:2 m'
    },
    # GFS: Global Forecast System
    'gfs': {
        'model': 'gfs', 
        'product': 'pgrb2.0p25', 
        'search': ':TMP:2 m'
    },
    # NAM: North American Mesoscale (12km)
    'nam': {
        'model': 'nam', 
        'product': 'conus', 
        'search': ':TMP:2 m'
    },
    # NAM Nest: 3km resolution
    # Note: 'search' is broad because NAM labels change
    'nam3k': {
        'model': 'nam', 
        'product': 'conusnest', 
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
