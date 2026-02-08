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


@st.cache_data
def precompute_grid(df: pd.DataFrame, nx: int, ny: int):
    # bbox
    x_min, x_max = -126, -66
    y_min, y_max = 24, 50

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, nx),
        np.linspace(y_min, y_max, ny)
    )

    points = df[['longitude', 'latitude']].values
    values = df['prob_snow'].values

    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    return (x_min, x_max, y_min, y_max, grid_x, grid_y, grid_z, points, values)


def build_base_figure(gdf: gpd.GeoDataFrame, grid_x, grid_y, grid_z, x_min, x_max, y_min, y_max):
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.subplots_adjust(top=0.85)

    levels = [0, 10, 25, 40, 50, 60, 75, 90, 100]
    color_list = [
        '#4f4f4d', '#6e7b8e', '#6080a8', '#6c97c4',
        '#82acd0', '#a1bddc', '#c5d0db', '#eeeeee'
    ]
    custom_cmap = mcolors.ListedColormap(color_list)
    norm = mcolors.BoundaryNorm(levels, custom_cmap.N)

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

    # state lines
    gdf.boundary.plot(ax=ax, color='black', linewidth=0.8, alpha=0.6, zorder=4)

    # labels
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
            p = row.geometry.representative_point()
            ax.annotate(
                text=row['short_name'],
                xy=(p.x, p.y),
                ha='center', va='center',
                color='white', fontsize=9, weight='bold',
                alpha=0.65,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black", alpha=0.5)],
                zorder=5
            )

    ax.text(
        x=-120, y=25,
        s="Probabilities for Alaska and\nHawaii not available",
        fontsize=10, color='#999999', ha='left', va='bottom', fontstyle='italic',
        zorder=6
    )

    plt.axis('off')
    fig.text(
        0.5, 0.92,
        "Historic probability of at least one inch of snow on Christmas",
        ha='center', va='bottom', fontsize=18, weight='bold', color='#333333'
    )

    cbar_ax = fig.add_axes([0.25, 0.88, 0.5, 0.02])
    cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal', ticks=levels)
    cbar.ax.tick_params(labelsize=10)

    return fig, ax, patch


def generate_weighted_flakes(points, values, x_min, x_max, y_min, y_max, n_flakes, seed):
    """
    Create initial flake positions weighted by probability.
    """
    rng = np.random.default_rng(seed)
    candidates = int(n_flakes * 6)

    xs = rng.uniform(x_min, x_max, size=candidates)
    ys = rng.uniform(y_min, y_max, size=candidates)

    probs = griddata(points, values, (xs, ys), method='linear')

    nan_mask = np.isnan(probs)
    if np.any(nan_mask):
        probs[nan_mask] = griddata(points, values, (xs[nan_mask], ys[nan_mask]), method='nearest')

    accept_p = np.clip(probs, 0, 100) / 100.0
    keep = rng.random(candidates) < accept_p

    fx = xs[keep]
    fy = ys[keep]

    # cap to n_flakes
    if fx.size > n_flakes:
        idx = rng.choice(fx.size, size=n_flakes, replace=False)
        fx, fy = fx[idx], fy[idx]

    # sizes + speed per flake (bigger flakes fall faster)
    sizes = rng.uniform(3, 18, size=fx.size)
    speed = rng.uniform(0.03, 0.12, size=fx.size) * (sizes / sizes.max())

    # alpha per flake
    alphas = rng.uniform(0.15, 0.60, size=fx.size)

    return fx, fy, sizes, speed, alphas


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Inputs")
    excel_filename = st.text_input(
        "Excel filename (repo root)",
        value="Christmas_day_snow_statistics_1991-2020_updated.xlsx"
    )

    quality = st.select_slider("Map resolution", options=["Low", "Medium", "High"], value="Medium")
    if quality == "Low":
        nx, ny = 700, 420
    elif quality == "High":
        nx, ny = 1400, 850
    else:
        nx, ny = 1000, 600

    st.header("Animated snow")
    snow_on = st.checkbox("Enable animation", value=True)
    snow_density = st.slider("Snow density", 0, 2000, 700, step=50)
    fps = st.slider("FPS (higher = smoother, heavier)", 2, 20, 10)
    wind = st.slider("Wind (left ↔ right)", -10, 10, 2)
    seed = st.number_input("Seed", 0, 9999, 7)

    colA, colB = st.columns(2)
    start_btn = colA.button("▶ Start")
    stop_btn = colB.button("⏹ Stop")


# ----------------------------
# Session state for animation
# ----------------------------
if "running" not in st.session_state:
    st.session_state.running = False

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False


# ----------------------------
# Load data
# ----------------------------
try:
    df = load_data(excel_filename)
except FileNotFoundError:
    st.error(f"Couldn't find **{excel_filename}** in the repo root.")
    st.stop()
except Exception as e:
    st.error(f"Excel load failed: {e}")
    st.stop()

try:
    gdf = load_states()
except Exception as e:
    st.error(f"Map load failed: {e}")
    st.stop()

x_min, x_max, y_min, y_max, grid_x, grid_y, grid_z, points, values = precompute_grid(df, nx, ny)

# ----------------------------
# Render (static or animated)
# ----------------------------
plot_area = st.empty()

if not snow_on or snow_density == 0:
    # just draw base map
    fig, ax, patch = build_base_figure(gdf, grid_x, grid_y, grid_z, x_min, x_max, y_min, y_max)
    plot_area.pyplot(fig, use_container_width=True)
    st.session_state.running = False
else:
    # Prepare flakes ONCE per run (so they move smoothly)
    # Re-generate when seed/density changes.
    key = (snow_density, seed, nx, ny)
    if "flake_key" not in st.session_state or st.session_state.flake_key != key:
        st.session_state.flake_key = key
        fx, fy, sizes, speed, alphas = generate_weighted_flakes(
            points, values, x_min, x_max, y_min, y_max,
            n_flakes=snow_density, seed=seed
        )
        st.session_state.fx = fx
        st.session_state.fy = fy
        st.session_state.sizes = sizes
        st.session_state.speed = speed
        st.session_state.alphas = alphas

    # draw one frame (even if not running)
    def draw_frame():
        fig, ax, clip_patch = build_base_figure(gdf, grid_x, grid_y, grid_z, x_min, x_max, y_min, y_max)

        fx = st.session_state.fx
        fy = st.session_state.fy
        sizes = st.session_state.sizes
        alphas = st.session_state.alphas

        colors = np.ones((fx.size, 4))
        colors[:, :3] = 1.0
        colors[:, 3] = alphas

        snow = ax.scatter(
            fx, fy,
            s=sizes,
            c=colors,
            marker='o',
            linewidths=0,
            zorder=10
        )
        snow.set_clip_path(clip_patch)

        plot_area.pyplot(fig, use_container_width=True)

    # If not running, show one frame
    if not st.session_state.running:
        draw_frame()
    else:
        # Animate loop (will run until Stop pressed)
        dt = 1.0 / fps
        for _ in range(5000):  # safety cap
            if not st.session_state.running:
                break

            # update flake positions
            fx = st.session_state.fx
            fy = st.session_state.fy
            sp = st.session_state.speed

            # fall + wind (wind scaled down)
            fy = fy - sp
            fx = fx + (wind * 0.005) * sp

            # wrap around
            fy = np.where(fy < y_min, y_max, fy)
            fx = np.where(fx < x_min, x_max, fx)
            fx = np.where(fx > x_max, x_min, fx)

            st.session_state.fx = fx
            st.session_state.fy = fy

            draw_frame()
            time.sleep(dt)

        st.session_state.running = False
