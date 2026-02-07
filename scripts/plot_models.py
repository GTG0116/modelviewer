import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie
from datetime import datetime, timedelta
import warnings
import os

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

def plot_temperature(model_name, config):
    print(f"--- Processing {model_name} ---")
    
    try:
        # 1. Initialize Herbie
        # We look for the latest run. If today's isn't in, it falls back automatically or we catch it.
        # RRFS is experimental, so we explicitly look for the control member.
        dt = datetime.utcnow()
        
        # Herbie arguments
        kwargs = {
            'model': config['model'],
            'product': config['product'],
            'save_dir': './data'
        }
        
        if 'member' in config:
            kwargs['member'] = config['member']

        # Get latest available run (check last 6 hours)
        H = Herbie(dt, **kwargs)
        
        # 2. Download/Select Data (2m Temperature)
        # Search string for GRIB2: 2m Temperature
        search_str = ":TMP:2 m above ground"
        
        # Download and open with Xarray
        # verbose=False keeps logs clean
        ds = H.xarray(search_str, verbose=True)
        
        # Extract variables (Herbie/cfgrib naming can vary slightly, usually 't2m')
        if 't2m' in ds:
            data = ds.t2m
        else:
            print(f"Variable 't2m' not found in {model_name}. Keys: {list(ds.keys())}")
            return

        # Convert Kelvin to Fahrenheit
        data_f = (data - 273.15) * 9/5 + 32

        # 3. Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, linewidth=0.5)

        # Plot data
        # Use a fixed range for better comparison (e.g., -20F to 110F)
        mesh = ax.pcolormesh(data.longitude, data.latitude, data_f,
                             transform=ccrs.PlateCarree(),
                             cmap='turbo', vmin=-10, vmax=100, shading='auto')

        plt.colorbar(mesh, orientation='horizontal', pad=0.05, label='Temperature (°F)')
        
        # Title with Model Run info
        valid_time = str(data.valid_time.values).split('.')[0]
        plt.title(f"{model_name.upper()} 2m Temperature\nValid: {valid_time} UTC", loc='left')
        
        # Save output
        out_path = f"images/{model_name}_temp.png"
        os.makedirs("images", exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', dpi=100)
        plt.close()
        print(f"Successfully saved {out_path}")

    except Exception as e:
        print(f"Failed to process {model_name}: {e}")

# Model Configurations
# Note: RRFS is experimental and hosted on AWS, it may occasionally fail if data is late.
models = {
    'hrrr':  {'model': 'hrrr', 'product': 'sfc'},
    'gfs':   {'model': 'gfs',  'product': 'pgrb2.0p25'},
    'nam':   {'model': 'nam',  'product': 'conus'},
    'nam3k': {'model': 'nam',  'product': 'conusnest'}, # NAM 3km Nest
    'rrfs':  {'model': 'rrfs', 'product': 'prslev', 'member': 'control'} # Experimental
}

if __name__ == "__main__":
    for name, config in models.items():
        plot_temperature(name, config)
