import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings("ignore")

def get_latest_run_time(model_name):
    """Snaps the current time to the most likely valid model cycle."""
    now = datetime.utcnow()
    if model_name == 'hrrr':
        # HRRR runs every hour; look at the top of the hour
        return now.replace(minute=0, second=0, microsecond=0)
    else:
        # GFS/NAM run every 6 hours (00, 06, 12, 18)
        hour = (now.hour // 6) * 6
        return now.replace(hour=hour, minute=0, second=0, microsecond=0)

def plot_temperature(model_name, config):
    print(f"\n--- Checking {model_name.upper()} ---")
    
    # Start from the most recent theoretical cycle
    base_time = get_latest_run_time(model_name)
    
    found_data = False
    # Check the last 4 available cycles
    for i in range(4):
        # For GFS/NAM, we go back 6 hours per step. For HRRR, 1 hour.
        decrement = 6 if model_name != 'hrrr' else 1
        search_dt = base_time - timedelta(hours=i * decrement)
        
        try:
            print(f"Searching for {model_name} cycle: {search_dt.strftime('%Y-%m-%d %H:%00 UTC')}")
            
            H = Herbie(
                search_dt,
                model=config['model'],
                product=config['product'],
                fxx=0, # We want the 'Analysis' (current state)
                priority=['aws', 'nomads'],
                save_dir='./data'
            )
            
            # Use a very specific search string for 2m Temp to avoid index errors
            # Herbie uses 't2m' or 'TMP:2 m' depending on the model
            ds = H.xarray("TMP:2 m above ground", verbose=False)
            
            if ds is None:
                continue

            # Standardize variable name (Herbie/cfgrib often maps TMP:2m to 't2m')
            data_var = 't2m' if 't2m' in ds else 't'
            data_f = (ds[data_var] - 273.15) * 9/5 + 32

            # --- Plotting ---
            fig = plt.figure(figsize=(12, 9))
            proj = ccrs.LambertConformal(central_longitude=-75, central_latitude=42)
            ax = fig.add_subplot(1, 1, 1, projection=proj)
            
            # Northeast Extent
            ax.set_extent([-82, -66, 37, 48], crs=ccrs.PlateCarree())

            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.8)
            ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')

            mesh = ax.pcolormesh(ds.longitude, ds.latitude, data_f,
                                 transform=ccrs.PlateCarree(),
                                 cmap='turbo', vmin=0, vmax=100, shading='auto')

            plt.colorbar(mesh, orientation='vertical', shrink=0.7, label='Temperature (°F)')
            
            plt.title(f"{model_name.upper()} 2m Temperature", loc='left', fontweight='bold')
            plt.title(f"Init: {search_dt.strftime('%H')}Z | Northeast Sector", loc='right')

            os.makedirs("images", exist_ok=True)
            out_path = f"images/{model_name}_temp.png"
            plt.savefig(out_path, bbox_inches='tight', dpi=120)
            plt.close()
            
            print(f"✅ Success! Generated {out_path}")
            found_data = True
            break 

        except Exception as e:
            # print(f"  (Skipping cycle {search_dt.hour}Z: {e})")
            continue
            
    if not found_data:
        print(f"❌ Could not find valid data for {model_name} in recent cycles.")

models = {
    'hrrr':  {'model': 'hrrr', 'product': 'sfc'},
    'gfs':   {'model': 'gfs',  'product': 'pgrb2.0p25'},
    'nam':   {'model': 'nam',  'product': 'conus'},
    'nam3k': {'model': 'nam',  'product': 'conusnest'}
}

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    for name, config in models.items():
        plot_temperature(name, config)
