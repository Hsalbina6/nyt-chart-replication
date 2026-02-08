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

# ----------------------------
# Helpers (cached)
# ----------------------------
@st.cache_data
def load_data(excel_filename: str) -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, excel_filename)
    
    try:
        df = pd.read_excel(file_path, header=6)
    except:
        return pd.DataFrame() # Return empty if fail

    prob_col = [c for c in df.columns if 'Probability' in str(c)][0]
    df = df.rename(columns={'Lat': 'latitude', 'Lon': 'longitude', prob_col: 'prob_snow'})
    df['prob_snow'] = pd.to_numeric(df['prob_snow'], errors='coerce')
    df = df[df['prob_snow'] != -9999]
    df = df.dropna(subset=['prob_snow', 'latitude', 'longitude'])

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

@st.cache_data
def compute_grid(df, nx=1000, ny=600):
    x_min, x_max, y_min, y_max = -126, -66, 24, 50
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    points = df[['longitude', 'latitude']].values
    values = df['prob_snow'].values
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    return grid_x, grid_y, grid_z

# ----------------------------
# The Plotting Logic
# ----------------------------
def make_figure(gdf, grid_x, grid_y, grid_z, threshold=0, show_snowflakes=False):
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.subplots_adjust(top=0.85)

    # --- 1. Probability Masking ---
    z_plot = grid_z.copy()
    if threshold > 0:
        z_plot = np.ma.masked_less(z_plot, threshold)

    # --- 2. Contour Plot ---
    levels = [0, 10, 25, 40, 50, 60, 75, 90, 100]
    color_list = ['#4f4f4d', '#6e7b8e', '#6080a8', '#6c97c4', '#82acd0', '#a1bddc', '#c5d0db', '#eeeeee']
    custom_cmap = mcolors.ListedColormap(color_list)
    norm = mcolors.BoundaryNorm(levels, custom_cmap.N)

    contour = ax.contourf(grid_x, grid_y, z_plot, levels=levels, cmap=custom_cmap, norm=norm, alpha=1.0, extend='neither')

    # --- 3. Clip to US ---
    usa_boundary = gdf.dissolve().geometry.iloc[0]
    polys = usa_boundary.geoms if hasattr(usa_boundary, 'geoms') else [usa_boundary]
    path = Path.make_compound_path(*[Path(np.array(poly.exterior.coords)) for poly in polys])
    patch = PathPatch(path, transform=ax.transData, facecolor='none')
    contour.set_clip_path(patch)

    # Draw State Lines
    gdf.boundary.plot(ax=ax, color='black', linewidth=0.8, alpha=0.6)

    # --- 4. Add State Names ---
    state_map = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
        'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
        'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
        'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }
    
    for _, row in gdf.iterrows():
        short_name = state_map.get(row['name'])
        if short_name:
            centroid = row.geometry.representative_point()
            ax.annotate(text=short_name, xy=(centroid.x, centroid.y), ha='center', va='center',
                        color='white', fontsize=9, weight='bold', alpha=0.7,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="black", alpha=0.5)])

    # --- 5. Snowflake Overlay (Animation 2) ---
    if show_snowflakes:
        # Generate random white dots within US Lat/Lon box
        # We generate 400 flakes per frame
        snow_x = np.random.uniform(-125, -66, 400)
        snow_y = np.random.uniform(24, 50, 400)
        ax.scatter(snow_x, snow_y, color='white', s=5, alpha=0.8, zorder=10)


    # --- 6. Titles & Text ---
    ax.text(x=-120, y=25, s="Probabilities for Alaska and\nHawaii not available",
            fontsize=10, color='#999999', ha='left', va='bottom', fontstyle='italic')

    plt.axis('off')
    
    # Dynamic Title
    title_text = "Historic probability of at least one inch of snow on Christmas"
    if threshold > 0:
        title_text += f"\n(Highlighting areas > {threshold}%)"
        
    fig.text(0.5, 0.92, title_text, ha='center', va='bottom', fontsize=18, weight='bold', color='#333333')
    
    cbar_ax = fig.add_axes([0.25, 0.88, 0.5, 0.02])
    fig.colorbar(contour, cax=cbar_ax, orientation='horizontal', ticks=levels)
    
    return fig

# ----------------------------
# Sidebar & Execution
# ----------------------------
excel_filename = "Christmas_day_snow_statistics_1991-2020_updated.xlsx"

# Page Layout
st.title("Interactive NYT Snow Map")
st.markdown("Use the controls on the left to animate the map.")

with st.sidebar:
    st.header("Animation Controls")
    anim_choice = st.radio("Choose Mode:", 
                          ["Static Map", "Animation 1: Snow Retreat", "Animation 2: Falling Snow"])
    
    if anim_choice == "Animation 1: Snow Retreat":
        speed = st.slider("Speed", 0.05, 0.5, 0.1)

# Load Data
df = load_data(excel_filename)
gdf = load_states()

if df.empty:
    st.error(f"Could not load '{excel_filename}'. Please ensure it is uploaded to GitHub.")
    st.stop()

gx, gy, gz = compute_grid(df)
placeholder = st.empty()

# --- Logic for Modes ---

if anim_choice == "Static Map":
    # Just show the standard map
    fig = make_figure(gdf, gx, gy, gz, threshold=0, show_snowflakes=False)
    placeholder.pyplot(fig, use_container_width=True)

elif anim_choice == "Animation 1: Snow Retreat":
    if st.button("Start Animation"):
        # Loop 0 to 95
        for t in range(0, 100, 5):
            fig = make_figure(gdf, gx, gy, gz, threshold=t, show_snowflakes=False)
            placeholder.pyplot(fig, use_container_width=True)
            time.sleep(speed)
            plt.close(fig) # Free memory
    else:
        # Show start state
        fig = make_figure(gdf, gx, gy, gz, threshold=0, show_snowflakes=False)
        placeholder.pyplot(fig, use_container_width=True)

elif anim_choice == "Animation 2: Falling Snow":
    if st.button("Start Snowing"):
        # Run 20 frames of snowing
        for i in range(20):
            fig = make_figure(gdf, gx, gy, gz, threshold=0, show_snowflakes=True)
            placeholder.pyplot(fig, use_container_width=True)
            time.sleep(0.1)
            plt.close(fig)
    else:
        st.info("Click start to see snow falling on the map.")
        fig = make_figure(gdf, gx, gy, gz, threshold=0, show_snowflakes=False)
        placeholder.pyplot(fig, use_container_width=True)
