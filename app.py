import os
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
# Helpers (cached)
# ----------------------------
@st.cache_data
def load_data(excel_filename: str) -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, excel_filename)

    df = pd.read_excel(file_path, header=6)

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


def make_figure(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    nx: int = 1000,
    ny: int = 600,
    snow_density: int = 0,
    snow_seed: int = 7
):
    # --- Interpolation (Clean) ---
    x_min, x_max = -126, -66
    y_min, y_max = 24, 50

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, nx),
        np.linspace(y_min, y_max, ny)
    )

    points = df[['longitude', 'latitude']].values
    values = df['prob_snow'].values
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.subplots_adjust(top=0.85)

    # Discrete Levels
    levels = [0, 10, 25, 40, 50, 60, 75, 90, 100]

    # Colors (your exact palette)
    color_list = [
        '#4f4f4d',
        '#6e7b8e',
        '#6080a8',
        '#6c97c4',
        '#82acd0',
        '#a1bddc',
        '#c5d0db',
        '#eeeeee'
    ]
    custom_cmap = mcolors.ListedColormap(color_list)
    norm = mcolors.BoundaryNorm(levels, custom_cmap.N)

    # Plot Heatmap
    contour = ax.contourf(
        grid_x, grid_y, grid_z,
        levels=levels, cmap=custom_cmap, norm=norm,
        alpha=1.0, extend='neither'
    )

    # Clip to US
    usa_boundary = gdf.dissolve().geometry.iloc[0]
    polys = usa_boundary.geoms if hasattr(usa_boundary, 'geoms') else [usa_boundary]
    path = Path.make_compound_path(*[Path(np.array(poly.exterior.coords)) for poly in polys])
    patch = PathPatch(path, transform=ax.transData, facecolor='none')
    contour.set_clip_path(patch)

    # ----------------------------
    # NEW: Weighted snow overlay (A2)
    # ----------------------------
    if snow_density > 0:
        rng = np.random.default_rng(snow_seed)

        # Generate candidate flakes (more candidates -> smoother probability weighting)
        candidates = int(snow_density * 6)

        xs = rng.uniform(x_min, x_max, size=candidates)
        ys = rng.uniform(y_min, y_max, size=candidates)

        # Evaluate probability at candidate points (same interpolation engine)
        probs = griddata(points, values, (xs, ys), method='linear')

        # Handle NaNs (outside convex hull). Fill using nearest.
        nan_mask = np.isnan(probs)
        if np.any(nan_mask):
            probs_nearest = griddata(points, values, (xs[nan_mask], ys[nan_mask]), method='nearest')
            probs[nan_mask] = probs_nearest

        # Convert to [0,1] acceptance probability
        accept_p = np.clip(probs, 0, 100) / 100.0

        # Sample flakes (weighted)
        keep = rng.random(candidates) < accept_p

        fx = xs[keep]
        fy = ys[keep]

        # Optionally cap so it never gets crazy heavy
        # (keeps Streamlit fast)
        max_flakes = int(snow_density * 1.6)
        if fx.size > max_flakes:
            idx = rng.choice(fx.size, size=max_flakes, replace=False)
            fx, fy = fx[idx], fy[idx]

        # Random sizes + slight alpha variation
        sizes = rng.uniform(3, 18, size=fx.size)
        alphas = rng.uniform(0.15, 0.55, size=fx.size)

        # Matplotlib doesn't allow per-point alpha directly in scatter easily,
        # so we embed alpha in RGBA colors.
        colors = np.ones((fx.size, 4))
        colors[:, :3] = 1.0  # white
        colors[:, 3] = alphas

        snow = ax.scatter(
            fx, fy,
            s=sizes,
            c=colors,
            marker='o',
            linewidths=0,
            zorder=3
        )
        snow.set_clip_path(patch)

    # Draw State Lines
    gdf.boundary.plot(ax=ax, color='black', linewidth=0.8, alpha=0.6, zorder=4)

    # Labels
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

    gdf2 = gdf.copy()
    gdf2['short_name'] = gdf2['name'].map(state_map)

    for _, row in gdf2.iterrows():
        if pd.notnull(row['short_name']):
            centroid = row.geometry.representative_point()
            ax.annotate(
                text=row['short_name'],
                xy=(centroid.x, centroid.y),
                ha='center', va='center',
                color='white', fontsize=9, weight='bold',
                alpha=0.65,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black", alpha=0.5)],
                zorder=5
            )

    # Disclaimer
    ax.text(
        x=-120, y=25,
        s="Probabilities for Alaska and\nHawaii not available",
        fontsize=10, color='#999999', ha='left', va='bottom', fontstyle='italic',
        zorder=6
    )

    # Title + Top colorbar
    plt.axis('off')
    fig.text(
        0.5, 0.92,
        "Historic probability of at least one inch of snow on Christmas",
        ha='center', va='bottom', fontsize=18, weight='bold', color='#333333'
    )

    cbar_ax = fig.add_axes([0.25, 0.88, 0.5, 0.02])
    cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal', ticks=levels)
    cbar.ax.tick_params(labelsize=10)

    return fig


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Inputs")
    excel_filename = st.text_input(
        "Excel filename (in repo root)",
        value="Christmas_day_snow_statistics_1991-2020_updated.xlsx"
    )

    quality = st.select_slider("Map resolution", options=["Low", "Medium", "High"], value="Medium")
    if quality == "Low":
        nx, ny = 700, 420
    elif quality == "High":
        nx, ny = 1400, 850
    else:
        nx, ny = 1000, 600

    st.header("Snow effect (A2)")
    snow_density = st.slider("Snow density", 0, 2500, 600, step=50)
    snow_seed = st.number_input("Snow pattern seed", min_value=0, max_value=9999, value=7, step=1)


# ----------------------------
# Load and render
# ----------------------------
try:
    df = load_data(excel_filename)
except FileNotFoundError:
    st.error(
        f"Couldn't find **{excel_filename}** in your repo.\n\n"
        "✅ Fix: upload the Excel to the repo root (same folder as app.py), "
        "or change the filename in the sidebar to match exactly."
    )
    st.stop()
except Exception as e:
    st.error(f"Excel load failed: {e}")
    st.stop()

try:
    gdf = load_states()
except Exception as e:
    st.error(f"Map load failed: {e}")
    st.stop()

fig = make_figure(df, gdf, nx=nx, ny=ny, snow_density=snow_density, snow_seed=snow_seed)
st.pyplot(fig, use_container_width=True)
