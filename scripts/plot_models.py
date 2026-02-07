import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie
from datetime import datetime, timedelta
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

def plot_temperature(model_name, config):
    print(f"\n--- Processing {model_name} (Source: AWS) ---")
    
    found_data = False
    # Models are updated at different times; check last 12 hours
    start_time = datetime.utcnow()
    
    for hours_back in range(0, 13):
        search_dt = start_time - timedelta(hours=hours_back)
        
        try:
            # Force Herbie to use AWS as the primary source
            H = Herbie(
                search_dt, 
                model=config['model'], 
                product=config['product'], 
                priority=['aws', 'nomads'], # AWS is first
                save_dir='./data'
            )
            
            # GRIB search string for 2m temperature
            search_str = ":TMP:2 m"
            ds = H.xarray(search_str, verbose=False)
            
            if ds is None: continue

            # Handle different variable naming conventions
            temp_var = 't2m' if 't2m' in ds else 't'
            data_f = (ds[temp_var] - 273.15) * 9/5 + 32

            # --- Plotting ---
            fig = plt.figure(figsize=(12, 9))
            # Using Lambert Conformal for a better "Northeast" look
            proj = ccrs.LambertConformal(central_longitude=-75, central_latitude=42)
            ax = fig.add_subplot(1, 1, 1, projection=proj)
            
            # SET NORTHEAST EXTENT [West, East, South, North]
            ax.set_extent([-82, -66, 37, 48], crs=ccrs.PlateCarree())

            # High-resolution features
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.8)
            ax.add_feature(cfeature.LAKES.with_scale('50m'), alpha=0.5)

            # Plot data
            # Transform is PlateCarree because GRIB coords are Lat/Lon
            mesh = ax.pcolormesh(ds.longitude, ds.latitude, data_f,
                                 transform=ccrs.PlateCarree(),
                                 cmap='turbo', vmin=0, vmax=100, shading='auto')

            plt.colorbar(mesh, orientation='vertical', shrink=0.7, label='Temp (°F)')
            
            # Title
            init_str = search_dt.strftime('%Y-%m-%d %H:%M UTC')
            plt.title(f"{model_name.upper()} 2m Temperature", loc='left', fontweight='bold')
            plt.title(f"Init: {init_str}\nNortheast Sector", loc='right', fontsize=10)

            os.makedirs("images", exist_ok=True)
            out_path = f"images/{model_name}_temp.png"
            plt.savefig(out_path, bbox_inches='tight', dpi=120)
            plt.close()
            
            print(f"✅ Success! Saved {out_path}")
            found_data = True
            break # Exit search loop for this model

        except Exception:
            continue
            
    if not found_data:
        print(f"❌ Failed to find {model_name} data on AWS/NOMADS for last 12h.")

# Active models
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
