import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings("ignore")

def get_latest_run_time(model_name):
    now = datetime.utcnow()
    if model_name == 'hrrr':
        return now.replace(minute=0, second=0, microsecond=0)
    else:
        # NAM and GFS run 00, 06, 12, 18
        hour = (now.hour // 6) * 6
        return now.replace(hour=hour, minute=0, second=0, microsecond=0)

def plot_temperature(model_name, config):
    print(f"\n--- Investigating {model_name.upper()} ---")
    
    base_time = get_latest_run_time(model_name)
    found_data = False
    
    # Check the last 6 available cycles (going back further for NAM)
    for i in range(6):
        decrement = 6 if model_name != 'hrrr' else 1
        search_dt = base_time - timedelta(hours=i * decrement)
        
        try:
            print(f"  Checking {search_dt.strftime('%Y-%m-%d %H:00')}...")
            
            H = Herbie(
                search_dt,
                model=config['model'],
                product=config['product'],
                fxx=0, 
                priority=['aws', 'nomads'],
                save_dir='./data'
            )
            
            # BROAD SEARCH: NAM sometimes uses '2 m above ground', others '2 m'
            # This regex-style search captures both.
            ds = H.xarray(":TMP:2 m", verbose=False)
            
            if ds is None:
                # One last try for NAM3k specifically: check f01 if f00 is missing
                if model_name == 'nam3k':
                    H = Herbie(search_dt, model='nam', product='conusnest', fxx=1)
                    ds = H.xarray(":TMP:2 m", verbose=False)
                
                if ds is None: continue

            # Standardize variable name
            # NAM often labels it 't' or 't2m'
            data_var = [v for v in ds.data_vars if 't' in v or 'TMP' in v][0]
            data_f = (ds[data_var] - 273.15) * 9/5 + 32

            # --- Plotting ---
            fig = plt.figure(figsize=(12, 9))
            proj = ccrs.LambertConformal(central_longitude=-75, central_latitude=42)
            ax = fig.add_subplot(1, 1, 1, projection=proj)
            
            # Northeast Sector
            ax.set_extent([-82, -66, 37, 48], crs=ccrs.PlateCarree())

            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.8)
            ax.add_feature(cfeature.LAKES.with_scale('50m'), alpha=0.3)

            mesh = ax.pcolormesh(ds.longitude, ds.latitude, data_f,
                                 transform=ccrs.PlateCarree(),
                                 cmap='turbo', vmin=0, vmax=100, shading='auto')

            plt.colorbar(mesh, orientation='vertical', shrink=0.7, label='Temperature (°F)')
            
            # Timestamps for verification
            plt.title(f"{model_name.upper()} 2m Temperature", loc='left', fontweight='bold')
            plt.title(f"Init: {search_dt.strftime('%m/%d %H')}Z | Sector: NE", loc='right')

            os.makedirs("images", exist_ok=True)
            out_path = f"images/{model_name}_temp.png"
            plt.savefig(out_path, bbox_inches='tight', dpi=120)
            plt.close()
            
            print(f"✅ SUCCESS: {model_name} generated at {out_path}")
            found_data = True
            break 

        except Exception as e:
            # Uncomment for local debugging:
            # print(f"    Debug: {e}")
            continue
            
    if not found_data:
        print(f"❌ FAIL: {model_name} not found after checking 6 cycles.")

# Updated Product Mapping for AWS/Herbie
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
