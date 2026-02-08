import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import griddata
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NYT Snow Map Replication", layout="wide")

# --- 1. Load Data (Cached) ---
@st.cache_data
def load_data():
    # NOTE: Ensure this file is uploaded to your GitHub repository!
    file_path = 'Christmas_day_snow_statistics_1991-2020_updated.xlsx'
    
    try:
        df = pd.read_excel(file_path, header=6)
    except Exception as e:
        st.error(f"Error loading file: {e}. Make sure the .xlsx file is in the GitHub repo.")
        return pd.DataFrame()

    # Clean Columns
    prob_col = [c for c in df.columns if 'Probability' in c][0]
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

df = load_data()

# --- 2. Load Map (Cached) ---
@st.cache_data
def load_map():
    map_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    gdf = gpd.read_file(map_url)
    gdf = gdf[~gdf['name'].isin(['Alaska', 'Hawaii', 'Puerto Rico'])]
    return gdf

gdf = load_map()

# --- 3. Sidebar Interactivity ---
st.sidebar.header("Map Controls")
min_prob = st.sidebar.slider("Highlight Areas with Probability > X%", 0, 100, 0)

# Filter data based on slider (optional visual interaction)
# For the main heatmap, we use all data, but we can display stats
filtered_stations = df[df['prob_snow'] >= min_prob]
st.sidebar.write(f"Stations matching criteria: {len(filtered_stations)}")

# --- 4. Plotting Logic ---
st.title("Historic Probability of White Christmas")
st.markdown("Replicating the New York Times visualization style.")

if not df.empty:
    # Interpolation Grid
    x_min, x_max = -126, -66
    y_min, y_max = 24, 50
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 1000),
        np.linspace(y_min, y_max, 600)
    )
    
    points = df[['longitude', 'latitude']].values
    values = df['prob_snow'].values
    
    # Linear interpolation
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

    # Create Plot
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Define Levels and Colors (NYT Style)
    levels = [0, 10, 25, 40, 50, 60, 75, 90, 100]
    color_list = [
        '#ffffff', '#e0e0e0', '#cccccc', '#99ccff', '#66b2ff', 
        '#3399ff', '#0080ff', '#0066cc', '#004c99'
    ]
    # Ensure color list matches level intervals (N levels - 1 colors needed, or N for boundary norm)
    # Adjusted to match your previous specific hex codes if preferred, 
    # but strictly mapping to the logic:
    cmap = mcolors.LinearSegmentedColormap.from_list("", color_list)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    # Plot Contour
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, norm=norm, alpha=1.0, extend='neither')

    # Clip to US Borders
    usa_boundary = gdf.dissolve().geometry.iloc[0]
    
    # Path Patch logic
    path = Path.make_compound_path(*[
        Path(np.array(poly.exterior.coords))
        for poly in (usa_boundary.geoms if hasattr(usa_boundary, 'geoms') else [usa_boundary])
    ])
    patch = PathPatch(path, transform=ax.transData, facecolor='none')
    contour.set_clip_path(patch)

    # Draw State Borders
    gdf.boundary.plot(ax=ax, color='white', linewidth=0.8, alpha=0.5)

    # State Labels
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
    gdf['short_name'] = gdf['name'].map(state_map)

    for idx, row in gdf.iterrows():
        if pd.notnull(row['short_name']):
            centroid = row.geometry.representative_point()
            ax.annotate(text=row['short_name'], xy=(centroid.x, centroid.y),
                        ha='center', va='center', color='black', fontsize=8, weight='bold')

    # Remove Axes
    ax.axis('off')

    # Render in Streamlit
    st.pyplot(fig)

else:
    st.warning("Data not loaded. Please ensure the Excel file is in the repository.")
