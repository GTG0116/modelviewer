import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie
from datetime import datetime, timedelta
import warnings
import os
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

def plot_temperature(model_name, config):
    print(f"\n--- Processing {model_name} ---")
    
    # Try looking back up to 12 hours to find the latest run
    # Models like NAM/GFS are 4-6 hours behind real-time on AWS
    found_data = False
    
    # We start searching from 2 hours ago to account for upload latency
    start_time = datetime.utcnow() - timedelta(hours=2)
    
    for hours_back in range(0, 13, 1): # Search previous 12 hours
        search_dt = start_time - timedelta(hours=hours_back)
        
        try:
            print(f"Checking {model_name} for run: {search_dt.strftime('%Y-%m-%d %H:%M')}")
            
            kwargs = {
                'model': config['model'],
                'product': config['product'],
                'save_dir': './data'
            }
            if 'member' in config:
                kwargs['member'] = config['member']

            # Initialize Herbie for this specific time
            H = Herbie(search_dt, **kwargs)
            
            # Search for 2m Temperature
            # We add a generic search to be safer across models
            search_str = ":(TMP|t):2 m above ground"
            
            # Attempt to download/access
            ds = H.xarray(search_str, verbose=False)
            
            if ds is None or (len(ds) == 0):
                continue
                
            # Normalize variable names (some models use t2m, some t)
            if 't2m' in ds:
                data = ds.t2m
            elif 't' in ds:
                data = ds.t
            else:
                continue

            # If we got here, we have data!
            print(f"Found data for {model_name}!")
            found_data = True
            
            # Convert K to F
            data_f = (data - 273.15) * 9/5 + 32

            # Plotting
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
            
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.STATES, linewidth=0.5)

            mesh = ax.pcolormesh(data.longitude, data.latitude, data_f,
                                 transform=ccrs.PlateCarree(),
                                 cmap='turbo', vmin=-10, vmax=110, shading='auto')

            plt.colorbar(mesh, orientation='horizontal', pad=0.05, label='Temperature (°F)')
            
            valid_str = str(data.valid_time.values).split('.')[0]
            plt.title(f"{model_name.upper()} 2m Temp | Init: {search_dt.strftime('%H')}Z", loc='left')
            plt.title(f"Valid: {valid_str}", loc='right')
            
            # Ensure directory exists
            os.makedirs("images", exist_ok=True)
            out_path = f"images/{model_name}_temp.png"
            plt.savefig(out_path, bbox_inches='tight', dpi=100)
            plt.close()
            print(f"Saved: {out_path}")
            
            # Stop looking back for this model
            break

        except Exception as e:
            # If it fails, we just try the next hour back
            continue
    
    if not found_data:
        print(f"Could not find data for {model_name} in the last 12 hours.")

models = {
    'hrrr':  {'model': 'hrrr', 'product': 'sfc'},
    'gfs':   {'model': 'gfs',  'product': 'pgrb2.0p25'},
    'nam':   {'model': 'nam',  'product': 'conus'},
    'nam3k': {'model': 'nam',  'product': 'conusnest'},
    # 'rrfs':  {'model': 'rrfs', 'product': 'prslev', 'member': 'control'} # Commented out RRFS as it is unstable
}

if __name__ == "__main__":
    # Ensure image directory exists before starting
    os.makedirs("images", exist_ok=True)
    
    for name, config in models.items():
        plot_temperature(name, config)
