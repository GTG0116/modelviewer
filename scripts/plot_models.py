import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie
from datetime import datetime, timedelta
import warnings
import os
import json
import sys

warnings.filterwarnings("ignore")

# Define the models and their specific complexities
models = {
    'hrrr':   {'model': 'hrrr', 'product': 'sfc', 'search': ':TMP:2 m', 'step': 1},
    'gfs':    {'model': 'gfs',  'product': 'pgrb2.0p25', 'search': ':TMP:2 m', 'step': 6},
    'nam':    {'model': 'nam',  'product': 'conus', 'search': ':TMP:2 m', 'step': 6},
    'nam3k':  {'model': 'nam',  'product': 'conusnest', 'search': ':TMP:2 m', 'step': 6},
    # RRFS is often delayed by 6-12 hours on the public bucket
    'rrfs':   {'model': 'rrfs', 'product': 'prslev', 'member': 'control', 'search': ':TMP:2 m', 'step': 1}
}

def get_start_time(model_name):
    """Returns the most recent possible run time for a model."""
    now = datetime.utcnow()
    if model_name in ['hrrr', 'rrfs']:
        return now.replace(minute=0, second=0, microsecond=0)
    else:
        # Synoptic models (GFS, NAM) run at 00, 06, 12, 18
        hour = (now.hour // 6) * 6
        return now.replace(hour=hour, minute=0, second=0, microsecond=0)

def plot_model(name, config):
    print(f"\n--- Processing {name.upper()} ---")
    base_time = get_start_time(name)
    
    # Look back up to 12 cycles to find data
    # (NAM and RRFS on AWS are notoriously slow to upload)
    for i in range(12):
        delta = i * config['step']
        run_dt = base_time - timedelta(hours=delta)
        
        try:
            print(f"  Checking {run_dt.strftime('%H')}z run...")
            
            # 1. Initialize Herbie
            # We strictly prioritize AWS, then NOMADS (NOAA)
            H = Herbie(
                run_dt,
                model=config['model'],
                product=config['product'],
                member=config.get('member'),
                priority=['aws', 'nomads'], 
                save_dir='./data'
            )

            # 2. Search for Data
            # "fxx=0" is the Analysis hour. 
            ds = H.xarray(config['search'], fxx=0, verbose=False)
            
            if ds is None or ds.sizes == {}:
                continue

            # 3. Variable Extraction
            # Models name things differently (t, t2m, etc.)
            var_name = list(ds.data_vars)[0]
            data = ds[var_name]
            
            # Convert K -> F
            data_f = (data - 273.15) * 9/5 + 32

            # 4. Plotting (Northeast US)
            fig = plt.figure(figsize=(10, 8))
            proj = ccrs.LambertConformal(central_longitude=-75, central_latitude=42)
            ax = fig.add_subplot(1, 1, 1, projection=proj)
            ax.set_extent([-82, -66, 37, 47], crs=ccrs.PlateCarree())

            # Features
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
            ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')

            # Mesh
            mesh = ax.pcolormesh(ds.longitude, ds.latitude, data_f,
                                 transform=ccrs.PlateCarree(),
                                 cmap='turbo', vmin=-10, vmax=100, shading='auto')
            
            plt.colorbar(mesh, orientation='vertical', shrink=0.7, label='Temperature (°F)', pad=0.02)
            
            # Labels
            init_str = run_dt.strftime('%Y-%m-%d %H')
            plt.title(f"{name.upper()}", loc='left', fontweight='bold', fontsize=14)
            plt.title(f"Init: {init_str}Z", loc='right', fontsize=10)

            # Save
            out_file = f"site/images/{name}.png"
            os.makedirs("site/images", exist_ok=True)
            plt.savefig(out_file, bbox_inches='tight', dpi=100)
            plt.close()
            
            print(f"  ✅ Saved {out_file}")
            
            # Return Metadata for the website
            return {
                "status": "success",
                "run": f"{init_str}Z",
                "image": f"images/{name}.png"
            }

        except Exception as e:
            # print(f"    Error: {e}") 
            continue

    print(f"  ❌ Failed to find data for {name}")
    return {"status": "failed", "image": "images/placeholder.png"}

if __name__ == "__main__":
    # Create site directory
    os.makedirs("site/images", exist_ok=True)
    
    status_report = {}
    
    for model_name, config in models.items():
        status_report[model_name] = plot_model(model_name, config)
    
    # Save the status JSON for the frontend to read
    with open("site/status.json", "w") as f:
        json.dump(status_report, f, indent=2)
