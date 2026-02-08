import os
import time
import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
from scipy.interpolate import griddata
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# ----------------------------
# Streamlit page setup
# ----------------------------
st.set_page_config(page_title="NYT Snow Map", layout="wide")
st.title("Historic probability of at least one inch of snow on Christmas")
st.caption("NOAA 1991–2020 Climate Normals • Replication-style map")

# ----------------------------
# 1. Load Data (Cached)
# ----------------------------
@st.cache_data
def load_data(excel_filename: str) -> pd.DataFrame:
    # Resolve file path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, excel_filename)

    try:
        df = pd.read_excel(file_path, header=6)
    except:
        return pd.DataFrame() # Graceful fail

    # Clean Columns
    prob_col = [c for c in df.columns if 'Probability' in str(c)][0]
    df = df.rename(columns={'Lat': 'latitude', 'Lon': 'longitude', prob_col: 'prob_snow'})
    df['prob_snow'] = pd.to_numeric(df['prob_snow'], errors='coerce')
    df = df[df['prob_snow'] != -9999]
    df = df.dropna(subset=['prob_snow', 'latitude', 'longitude'])

    # Filter for Continental US
    df = df[
        (df['latitude'] >= 24) & (df['latitude'] <= 50) &
        (df['longitude'] >= -125) & (df['longitude'] <= -66)
    ]
    return df

@st.cache_resource
def load_states() -> gpd.GeoDataFrame:
    map_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    gdf = gpd.read_file(map_url)
    gdf = gdf[~gdf['name'].isin(['Alaska', 'Hawaii', 'Puerto Rico'])]
    return gdf

# ----------------------------
# 2. Compute Grid (Cached for Speed)
# ----------------------------
@st.cache_data
def compute_grid(df, nx=1000, ny=600):
    """
    Performs the heavy lifting (interpolation) ONCE.
    Returns grid_x, grid_y, and grid_z.
    """
    x_min, x_max = -126, -66
    y_min, y_max = 24, 50

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, nx),
        np.linspace(y_min, y_max, ny)
    )

    points = df[['longitude', 'latitude']].values
    values = df['prob_snow'].values
    
    # Linear interpolation
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    
    return grid_x, grid_y, grid_z

# ----------------------------
# 3. Plotting Function
# ----------------------------
def plot_frame(gdf, grid_x, grid_y, grid_z, threshold=0):
    """
    Draws a single frame. 
    If threshold > 0, it masks out areas with low probability.
    """
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.subplots_adjust(top=0.85)

    # --- Threshold Logic for Animation ---
    # Create a copy of grid_z to mask
    z_to_plot = grid_z.copy()
    
    if threshold > 0:
        # Mask values below the threshold so they appear white/empty
        z_to_plot = np.ma.masked_less(z_to_plot, threshold)

    # Discrete Levels (NYT Style)
    levels = [0, 10, 25, 40, 50, 60, 75, 90, 100]
    color_list = [
        '#4f4f4d', '#6e7b8e', '#6080a8', '#6c97c4', 
        '#82acd0', '#a1bddc', '#c5d0db', '#eeeeee'
    ]
    custom_cmap = mcolors.ListedColormap(color_list)
    norm = mcolors.BoundaryNorm(levels, custom_cmap.N)

    # Plot Heatmap
    contour = ax.contourf(
        grid_x, grid_y, z_to_plot,
        levels=levels, cmap=custom_cmap, norm=norm,
        alpha=1.0, extend='neither'
    )

    # Clip to US Boundary
    usa_boundary = gdf.dissolve().geometry.iloc[0]
    polys = usa_boundary.geoms if hasattr(usa_boundary, 'geoms') else [usa_boundary]
    path = Path.make_compound_path(*[Path(np.array(poly.exterior.coords)) for poly in polys])
    patch = PathPatch(path, transform=ax.transData, facecolor='none')
    contour.set_clip_path(patch)

    # Draw State Lines
    gdf.boundary.plot(ax=ax, color='black', linewidth=0.8, alpha=0.6)

    # Clean up axes
    plt.axis('off')
    
    # Title showing current state
    header_text = "Historic probability of at least one inch of snow on Christmas"
    if threshold > 0:
        header_text += f"\n(Highlighting areas > {threshold}%)"
        
    fig.text(0.5, 0.92, header_text, ha='center', va='bottom', fontsize=18, weight='bold', color='#333333')

    # Colorbar (Static position)
    cbar_ax = fig.add_axes([0.25, 0.88, 0.5, 0.02])
    cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal', ticks=levels)
    cbar.ax.tick_params(labelsize=10)

    return fig

# ----------------------------
# 4. Main App Logic
# ----------------------------
# Sidebar Controls
with st.sidebar:
    st.header("Controls")
    excel_filename = st.text_input("Excel filename", value="Christmas_day_snow_statistics_1991-2020_updated.xlsx")
    
    mode = st.radio("View Mode", ["Static Map", "Animation"])
    
    if mode == "Static Map":
        st.info("Displays the standard full map.")
    else:
        st.info("Animates probability thresholds from 0% to 90%.")
        speed = st.slider("Animation Speed", 0.05, 1.0, 0.1)

# Load Data
try:
    df = load_data(excel_filename)
    gdf = load_states()
    
    if df.empty:
        st.error(f"Could not load data. check filename: {excel_filename}")
        st.stop()
        
    # Pre-calculate Grid (Important for animation speed)
    gx, gy, gz = compute_grid(df)

except Exception as e:
    st.error(f"Error initializing data: {e}")
    st.stop()

# Render based on mode
map_placeholder = st.empty()

if mode == "Static Map":
    fig = plot_frame(gdf, gx, gy, gz, threshold=0)
    map_placeholder.pyplot(fig, use_container_width=True)

elif mode == "Animation":
    if st.button("Start Animation"):
        # Loop through thresholds: 0, 5, 10, ... 95
        for t in range(0, 100, 5):
            fig = plot_frame(gdf, gx, gy, gz, threshold=t)
            map_placeholder.pyplot(fig, use_container_width=True)
            time.sleep(speed) # Pause to create animation frame
            plt.close(fig)    # Free memory



