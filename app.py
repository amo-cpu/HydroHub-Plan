import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import zipfile
import io
import json
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="HydroHub AI")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# SAFE API KEY — won't crash if secret missing
# ─────────────────────────────────────────────
try:
    ORS_API_KEY = st.secrets["ORS_API_KEY"]
except Exception:
    ORS_API_KEY = None


# ─────────────────────────────────────────────
# DATA LOADING
# Three sources tried in order:
#   1. Local cached CSV
#   2. SimpleMaps ZIP download (with validation)
#   3. Synthetic fallback (always works offline)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading ZIP code dataset…")
def build_datasets():
    zip_csv = os.path.join(DATA_DIR, "uszips.csv")

    # ── Try to load from cached file ──────────────────────────────────
    if os.path.exists(zip_csv):
        try:
            df_zip = pd.read_csv(zip_csv, dtype={"zip": str})
            df_zip = _normalize_zip_df(df_zip)
            if len(df_zip) > 100:
                return _enrich(df_zip)
        except Exception:
            pass  # fall through to download

    # ── Try SimpleMaps download ────────────────────────────────────────
    SIMPLEMAPS_URL = (
        "https://simplemaps.com/static/data/us-zips/1.79/basic/"
        "simplemaps_uszips_basicv1.79.zip"
    )
    try:
        resp = requests.get(SIMPLEMAPS_URL, timeout=20)
        resp.raise_for_status()

        # Validate it's actually a zip before opening
        if not resp.content[:4] == b'PK\x03\x04':
            raise ValueError(
                f"Response is not a ZIP file "
                f"(HTTP {resp.status_code}, {len(resp.content)} bytes). "
                "SimpleMaps may require registration."
            )

        z = zipfile.ZipFile(io.BytesIO(resp.content))
        # Find the CSV inside the zip
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV found inside downloaded ZIP")

        with z.open(csv_names[0]) as f:
            df_zip = pd.read_csv(f, dtype={"zip": str})

        df_zip = _normalize_zip_df(df_zip)
        df_zip.to_csv(zip_csv, index=False)   # cache locally
        return _enrich(df_zip)

    except Exception as e:
        st.warning(
            f"Could not download SimpleMaps data ({e}). "
            "Using built-in synthetic dataset instead. "
            "To use real data, place `uszips.csv` in the `data/` folder."
        )

    # ── Synthetic fallback — always works ─────────────────────────────
    return _build_synthetic()


def _normalize_zip_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names regardless of source."""
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {
        "zip":          "ZIP",
        "zipcode":      "ZIP",
        "zip_code":     "ZIP",
        "postal_code":  "ZIP",
        "lat":          "Latitude",
        "latitude":     "Latitude",
        "lng":          "Longitude",
        "lon":          "Longitude",
        "longitude":    "Longitude",
        "population":   "Population",
        "state_id":     "State",
        "state":        "State",
        "county_name":  "County",
        "county":       "County",
        "city":         "City",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Drop rows missing lat/lon
    for col in ("Latitude", "Longitude"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found after rename")
    df = df.dropna(subset=["Latitude", "Longitude"])

    if "ZIP" not in df.columns:
        df["ZIP"] = df.index.astype(str)
    if "Population" not in df.columns:
        df["Population"] = np.random.randint(1_000, 50_000, len(df))
    if "County" not in df.columns:
        df["County"] = "Unknown"
    if "State" not in df.columns:
        df["State"] = "Unknown"
    if "City" not in df.columns:
        df["City"] = "Unknown"

    df["ZIP"] = df["ZIP"].astype(str).str.zfill(5)
    return df


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk columns to a normalised ZIP dataframe."""
    np.random.seed(42)
    n = len(df)

    # Flood risk — higher near coasts and Gulf
    lon = df["Longitude"].values
    lat = df["Latitude"].values
    coastal_east  = np.clip((lon + 75) / 10, 0, 1)   # Atlantic coast
    coastal_gulf  = np.clip((lat - 25) / 8, 0, 1) * np.clip((-lon - 85) / 15, 0, 1)
    coastal_west  = np.clip((-lon - 118) / 8, 0, 1)  # Pacific coast
    df["FloodRisk"]     = np.clip(coastal_east * 0.4 + coastal_gulf * 0.5
                                  + coastal_west * 0.3
                                  + np.random.uniform(0, 0.3, n), 0, 1)
    df["HurricaneRisk"] = np.clip(coastal_gulf * 0.8 + coastal_east * 0.3
                                  + np.random.uniform(0, 0.2, n), 0, 1)
    df["CoastalRisk"]   = np.clip(coastal_east * 0.5 + coastal_gulf * 0.6
                                  + coastal_west * 0.4
                                  + np.random.uniform(0, 0.2, n), 0, 1)
    df["HistoricalDamage"] = np.random.randint(5_000, 2_000_000, n)
    return df


def _build_synthetic() -> pd.DataFrame:
    """
    Pure-Python synthetic dataset covering all 50 US states.
    No downloads, no external files. Used as offline fallback.
    """
    np.random.seed(42)

    # Representative cities with real coordinates
    CITIES = [
        ("10001","New York","NY","New York",40.748,-73.997,8_336_817),
        ("90001","Los Angeles","CA","Los Angeles",34.052,-118.244,3_979_576),
        ("60601","Chicago","IL","Cook",41.878,-87.630,2_693_976),
        ("77001","Houston","TX","Harris",29.760,-95.370,2_304_580),
        ("85001","Phoenix","AZ","Maricopa",33.448,-112.074,1_608_139),
        ("19101","Philadelphia","PA","Philadelphia",39.953,-75.165,1_603_797),
        ("78201","San Antonio","TX","Bexar",29.424,-98.494,1_434_625),
        ("92101","San Diego","CA","San Diego",32.716,-117.161,1_386_932),
        ("75201","Dallas","TX","Dallas",32.777,-96.797,1_304_379),
        ("78701","Austin","TX","Travis",30.267,-97.743,961_855),
        ("32099","Jacksonville","FL","Duval",30.332,-81.656,949_611),
        ("94101","San Francisco","CA","San Francisco",37.775,-122.419,881_549),
        ("43201","Columbus","OH","Franklin",39.961,-82.999,905_748),
        ("28201","Charlotte","NC","Mecklenburg",35.227,-80.843,885_708),
        ("46201","Indianapolis","IN","Marion",39.768,-86.158,876_862),
        ("98101","Seattle","WA","King",47.606,-122.332,753_675),
        ("80201","Denver","CO","Denver",39.739,-104.990,727_211),
        ("37201","Nashville","TN","Davidson",36.163,-86.782,689_447),
        ("73101","Oklahoma City","OK","Oklahoma",35.468,-97.516,681_054),
        ("89701","Las Vegas","NV","Clark",36.170,-115.140,651_319),
        ("97201","Portland","OR","Multnomah",45.505,-122.675,652_503),
        ("21201","Baltimore","MD","Baltimore",39.290,-76.612,593_490),
        ("53201","Milwaukee","WI","Milwaukee",43.039,-87.907,590_157),
        ("87101","Albuquerque","NM","Bernalillo",35.084,-106.650,560_218),
        ("85701","Tucson","AZ","Pima",32.223,-110.975,548_073),
        ("93701","Fresno","CA","Fresno",36.738,-119.787,542_107),
        ("95814","Sacramento","CA","Sacramento",38.582,-121.494,513_624),
        ("64101","Kansas City","MO","Jackson",39.100,-94.579,508_090),
        ("30301","Atlanta","GA","Fulton",33.749,-84.388,498_715),
        ("68101","Omaha","NE","Douglas",41.257,-95.935,486_051),
        ("33101","Miami","FL","Miami-Dade",25.762,-80.192,470_914),
        ("55401","Minneapolis","MN","Hennepin",44.978,-93.265,429_606),
        ("74101","Tulsa","OK","Tulsa",36.154,-95.993,413_066),
        ("27601","Raleigh","NC","Wake",35.780,-78.638,474_069),
        ("23201","Richmond","VA","Richmond",37.541,-77.436,230_436),
        ("02101","Boston","MA","Suffolk",42.360,-71.059,692_600),
        ("70112","New Orleans","LA","Orleans",29.951,-90.072,383_997),
        ("77550","Galveston","TX","Galveston",29.301,-94.798,50_180),
        ("28401","Wilmington","NC","New Hanover",34.226,-77.945,123_784),
        ("29401","Charleston","SC","Charleston",32.777,-79.931,150_227),
        ("33601","Tampa","FL","Hillsborough",27.951,-82.457,399_700),
        ("33401","West Palm Beach","FL","Palm Beach",26.715,-80.053,111_955),
        ("32801","Orlando","FL","Orange",28.538,-81.379,307_573),
        ("36101","Montgomery","AL","Montgomery",32.367,-86.300,199_518),
        ("36601","Mobile","AL","Mobile",30.695,-88.040,187_041),
        ("39201","Jackson","MS","Hinds",32.299,-90.185,153_701),
        ("70801","Baton Rouge","LA","East Baton Rouge",30.452,-91.187,225_374),
        ("77002","Beaumont","TX","Jefferson",30.080,-94.127,117_796),
        ("23501","Norfolk","VA","Norfolk",36.851,-76.286,244_703),
        ("21401","Annapolis","MD","Anne Arundel",38.978,-76.492,39_474),
        ("29501","Myrtle Beach","SC","Horry",33.689,-78.887,34_695),
        ("31401","Savannah","GA","Chatham",32.084,-81.100,147_088),
        ("99501","Anchorage","AK","Anchorage",61.218,-149.900,291_247),
        ("96801","Honolulu","HI","Honolulu",21.307,-157.858,350_964),
        ("04101","Portland","ME","Cumberland",43.659,-70.257,68_408),
        ("02901","Providence","RI","Providence",41.824,-71.413,190_934),
        ("06101","Hartford","CT","Hartford",41.766,-72.685,121_054),
        ("03101","Manchester","NH","Hillsborough",42.996,-71.455,115_644),
        ("05401","Burlington","VT","Chittenden",44.476,-73.212,45_012),
        ("19801","Wilmington","DE","New Castle",39.745,-75.548,70_898),
        ("08101","Camden","NJ","Camden",39.926,-75.120,73_562),
        ("82001","Cheyenne","WY","Laramie",41.140,-104.820,65_132),
        ("58501","Bismarck","ND","Burleigh",46.808,-100.784,73_529),
        ("57501","Pierre","SD","Hughes",44.368,-100.351,14_003),
        ("59601","Helena","MT","Lewis and Clark",46.596,-112.027,32_315),
        ("83701","Boise","ID","Ada",43.615,-116.202,235_684),
        ("84101","Salt Lake City","UT","Salt Lake",40.761,-111.891,200_591),
        ("89501","Reno","NV","Washoe",39.530,-119.814,250_998),
        ("98501","Olympia","WA","Thurston",47.038,-122.905,52_555),
        ("66101","Kansas City","KS","Wyandotte",39.116,-94.627,156_607),
        ("66801","Emporia","KS","Lyon",38.404,-96.181,24_916),
        ("74401","Muskogee","OK","Muskogee",35.748,-95.360,37_331),
        ("72201","Little Rock","AR","Pulaski",34.747,-92.290,202_591),
        ("65101","Jefferson City","MO","Cole",38.577,-92.173,43_330),
        ("62701","Springfield","IL","Sangamon",39.782,-89.650,114_230),
        ("47101","New Albany","IN","Floyd",38.286,-85.824,37_841),
        ("40601","Frankfort","KY","Franklin",38.200,-84.873,28_626),
        ("37601","Kingsport","TN","Sullivan",36.549,-82.562,54_564),
        ("39401","Hattiesburg","MS","Forrest",31.327,-89.291,47_020),
        ("36201","Anniston","AL","Calhoun",33.660,-85.831,21_606),
        ("32601","Gainesville","FL","Alachua",29.652,-82.325,133_997),
        ("34201","Bradenton","FL","Manatee",27.499,-82.575,57_834),
        ("29201","Columbia","SC","Richland",34.001,-81.035,133_451),
        ("27101","Winston-Salem","NC","Forsyth",36.100,-80.244,249_545),
        ("24501","Lynchburg","VA","Lynchburg",37.414,-79.142,82_168),
        ("25301","Charleston","WV","Kanawha",38.349,-81.633,46_692),
        ("17101","Harrisburg","PA","Dauphin",40.274,-76.884,50_099),
        ("14601","Rochester","NY","Monroe",43.157,-77.609,206_284),
        ("13201","Syracuse","NY","Onondaga",43.048,-76.147,142_327),
        ("12201","Albany","NY","Albany",42.653,-73.756,97_279),
        ("06901","Stamford","CT","Fairfield",41.053,-73.539,135_470),
        ("01101","Springfield","MA","Hampden",42.102,-72.590,153_984),
        ("01801","Woburn","MA","Middlesex",42.479,-71.152,40_117),
        ("07101","Newark","NJ","Essex",40.736,-74.172,311_549),
        ("20001","Washington","DC","District of Columbia",38.907,-77.037,689_545),
    ]

    rows = []
    for (zipcode, city, state, county, lat, lon, pop) in CITIES:
        # Add main city
        rows.append({
            "ZIP": zipcode, "City": city, "State": state,
            "County": county, "Latitude": lat, "Longitude": lon,
            "Population": pop
        })
        # Add 8 synthetic suburbs around each city
        for j in range(8):
            angle = j * 45 * np.pi / 180
            dist  = np.random.uniform(0.3, 1.2)
            rows.append({
                "ZIP":       str(int(zipcode) + j + 1).zfill(5),
                "City":      city,
                "State":     state,
                "County":    county,
                "Latitude":  lat  + dist * np.sin(angle),
                "Longitude": lon  + dist * np.cos(angle),
                "Population":np.random.randint(5_000, 80_000),
            })

    df = pd.DataFrame(rows)
    return _enrich(df)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = build_datasets()


# ─────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────
st.sidebar.title("🌊 HydroHub Controls")

hub_count   = st.sidebar.slider("Number of Hubs", 3, 60, 15)
flood_mult  = st.sidebar.slider("Flood Weight",        0.1, 3.0, 1.0)
hurr_mult   = st.sidebar.slider("Hurricane Weight",    0.1, 3.0, 1.0)
coastal_mult= st.sidebar.slider("Coastal Storm Weight",0.1, 3.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 ZIP / City Lookup")
zip_lookup  = st.sidebar.text_input("Enter ZIP code or city name")

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Disaster Scenario")
flood_scenario    = st.sidebar.slider("Flood Severity",    0.0, 2.0, 1.0)
hurricane_scenario= st.sidebar.slider("Hurricane Severity",0.0, 2.0, 1.0)

top_n = st.sidebar.slider("Top N Recommended Hubs", 1, 20, 10)


# ─────────────────────────────────────────────
# COMPUTE WEIGHTED RISK
# ─────────────────────────────────────────────
df = df.copy()
df["RiskWeight"] = (
    df["Population"] * (
        flood_mult   * flood_scenario    * df["FloodRisk"]
      + hurr_mult    * hurricane_scenario* df["HurricaneRisk"]
      + coastal_mult *                     df["CoastalRisk"]
    ) * (1 + df["HistoricalDamage"] / 1e6)
).clip(lower=0)


# ─────────────────────────────────────────────
# HUB OPTIMIZATION  (weighted k-means)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Optimizing hub locations…")
def optimize_hubs(lats, lons, weights, k):
    coords  = np.column_stack([lats, lons])
    w_norm  = weights / (weights.sum() + 1e-9)
    # Weighted resampling so k-means centres on high-risk areas
    idx     = np.random.choice(len(coords), size=min(20_000, len(coords)*5),
                               p=w_norm, replace=True)
    km = KMeans(n_clusters=k, n_init=15, random_state=42, max_iter=500)
    km.fit(coords[idx])
    hubs = pd.DataFrame(km.cluster_centers_, columns=["Latitude","Longitude"])
    hubs["HubID"] = range(len(hubs))
    return hubs

hubs = optimize_hubs(
    df["Latitude"].values, df["Longitude"].values,
    df["RiskWeight"].values, hub_count
)


# ─────────────────────────────────────────────
# ASSIGN NEAREST HUB (haversine-based)
# ─────────────────────────────────────────────
def haversine_matrix(lat1, lon1, lat2, lon2):
    """Vectorised haversine — returns miles."""
    R = 3958.8
    φ1 = np.radians(lat1)[:, None]
    φ2 = np.radians(lat2)[None, :]
    dφ = φ2 - φ1
    dλ = np.radians(lon2 - lon1)[None, :] - np.radians(np.zeros(len(lon1)))[:, None]
    a  = np.sin(dφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

dist_matrix = haversine_matrix(
    df["Latitude"].values, df["Longitude"].values,
    hubs["Latitude"].values, hubs["Longitude"].values
)
df["NearestHub"]      = dist_matrix.argmin(axis=1)
df["DistanceMiles"]   = dist_matrix.min(axis=1)
df["TravelMinutes"]   = df["DistanceMiles"] / 55 * 60 + 15   # drive time + staging


# ─────────────────────────────────────────────
# COVERAGE METRICS
# ─────────────────────────────────────────────
coverage = df.groupby("NearestHub").agg(
    PopulationCovered = ("Population",     "sum"),
    AvgDistanceMiles  = ("DistanceMiles",  "mean"),
    AvgTravelMinutes  = ("TravelMinutes",  "mean"),
    RiskExposure      = ("RiskWeight",     "sum"),
    ZIPsCovered       = ("ZIP",            "count"),
).reset_index()

# Merge hub lat/lon back
coverage = coverage.merge(hubs, on="HubID", how="left")


# ─────────────────────────────────────────────
# TOP RECOMMENDED LOCATIONS
# ─────────────────────────────────────────────
df["HubScore"] = df["RiskWeight"] / (df["TravelMinutes"] + 1)
top_recommended = (
    df.groupby(["Latitude","Longitude","City","State","ZIP"])
    .agg(TotalScore=("HubScore","sum"), PopulationCovered=("Population","sum"))
    .reset_index()
    .sort_values("TotalScore", ascending=False)
    .head(top_n)
)


# ─────────────────────────────────────────────
# DASHBOARD HEADER
# ─────────────────────────────────────────────
st.title("🌊 HydroHub AI — Emergency Response Optimizer")
st.caption("Flood · Hurricane · Coastal Storm · Real-time hub placement")

c1, c2, c3, c4 = st.columns(4)
c1.metric("ZIPs Modeled",           f"{len(df):,}")
c2.metric("Population Modeled",     f"{df['Population'].sum():,.0f}")
c3.metric("Avg Travel Time",        f"{df['TravelMinutes'].mean():.0f} min")
c4.metric("High-Risk ZIPs (top 10%)",
          f"{(df['RiskWeight'] > df['RiskWeight'].quantile(0.9)).sum():,}")


# ─────────────────────────────────────────────
# ZIP / CITY LOOKUP RESULTS
# ─────────────────────────────────────────────
lookup_result = None
if zip_lookup.strip():
    q = zip_lookup.strip()
    # Try ZIP match first
    match = df[df["ZIP"] == q.zfill(5)]
    # Then city name
    if match.empty:
        match = df[df["City"].str.contains(q, case=False, na=False)]
    # Then state
    if match.empty:
        match = df[df["State"].str.upper() == q.upper()]

    if not match.empty:
        lookup_result = match.iloc[0]
        st.success(f"📍 **{lookup_result.get('City','?')}, {lookup_result.get('State','?')}** — ZIP {lookup_result['ZIP']}")
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Flood Risk",    f"{lookup_result['FloodRisk']:.2f}")
        lc2.metric("Hurricane Risk",f"{lookup_result['HurricaneRisk']:.2f}")
        lc3.metric("Coastal Risk",  f"{lookup_result['CoastalRisk']:.2f}")
        lc4.metric("Nearest Hub",   f"Hub {int(lookup_result['NearestHub'])}")
        lc1.metric("Distance",      f"{lookup_result['DistanceMiles']:.1f} mi")
        lc2.metric("Travel Time",   f"{lookup_result['TravelMinutes']:.0f} min")
        lc3.metric("Population",    f"{int(lookup_result['Population']):,}")
        lc4.metric("Risk Score",    f"{lookup_result['RiskWeight']:,.0f}")
    else:
        st.warning(f"No match found for '{q}'. Try a 5-digit ZIP or city name.")


# ─────────────────────────────────────────────
# FOLIUM MAP
# ─────────────────────────────────────────────
center_lat = lookup_result["Latitude"] if lookup_result is not None else 39.0
center_lon = lookup_result["Longitude"] if lookup_result is not None else -98.0
zoom       = 10 if lookup_result is not None else 4

m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="CartoDB dark_matter")

# ── Risk heatmap ──────────────────────────────
heat_data = (
    df[["Latitude","Longitude","RiskWeight"]]
    .dropna()
    .values
    .tolist()
)
HeatMap(heat_data, radius=12, blur=18, max_zoom=10,
        gradient={0.0:"blue",0.4:"cyan",0.6:"yellow",0.8:"orange",1.0:"red"}
        ).add_to(m)

# ── City dots (sampled) ───────────────────────
cluster_layer = MarkerCluster(name="ZIP/City Nodes").add_to(m)
sample = df.sample(min(3000, len(df)), random_state=42)
risk_max = df["RiskWeight"].max() or 1

for _, row in sample.iterrows():
    intensity = int(min(255, row["RiskWeight"] / risk_max * 255))
    color = f"#{intensity:02x}{(255-intensity)//2:02x}00"
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.6,
        popup=folium.Popup(
            f"<b>{row.get('City','?')}, {row.get('State','?')}</b><br>"
            f"ZIP: {row['ZIP']}<br>"
            f"Population: {int(row['Population']):,}<br>"
            f"Flood Risk: {row['FloodRisk']:.2f}<br>"
            f"Nearest Hub: {int(row['NearestHub'])}<br>"
            f"Travel: {row['TravelMinutes']:.0f} min",
            max_width=200
        ),
    ).add_to(cluster_layer)

# ── Optimized hub markers ──────────────────────
for _, hub in hubs.iterrows():
    cov = coverage[coverage["NearestHub"] == hub["HubID"]]
    pop_cov = int(cov["PopulationCovered"].values[0]) if len(cov) else 0
    avg_min = float(cov["AvgTravelMinutes"].values[0]) if len(cov) else 0
    folium.Marker(
        location=[hub["Latitude"], hub["Longitude"]],
        icon=folium.Icon(color="green", icon="star", prefix="fa"),
        popup=folium.Popup(
            f"<b>Hub {int(hub['HubID'])}</b><br>"
            f"Pop Covered: {pop_cov:,}<br>"
            f"Avg Travel: {avg_min:.0f} min",
            max_width=180
        ),
        tooltip=f"Hub {int(hub['HubID'])}",
    ).add_to(m)

# ── 200-mile coverage circles ─────────────────
for _, hub in hubs.iterrows():
    folium.Circle(
        location=[hub["Latitude"], hub["Longitude"]],
        radius=322_000,   # 200 miles in metres
        color="cyan",
        weight=1,
        fill=True,
        fill_opacity=0.03,
    ).add_to(m)

# ── Recommended hub markers ───────────────────
for _, row in top_recommended.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        icon=folium.Icon(color="purple", icon="bolt", prefix="fa"),
        popup=folium.Popup(
            f"<b>Recommended Hub</b><br>"
            f"{row.get('City','?')}, {row.get('State','?')}<br>"
            f"Score: {row['TotalScore']:,.0f}<br>"
            f"Pop: {int(row['PopulationCovered']):,}",
            max_width=180
        ),
        tooltip="Recommended Hub",
    ).add_to(m)

# ── Lookup pin ────────────────────────────────
if lookup_result is not None:
    folium.Marker(
        location=[lookup_result["Latitude"], lookup_result["Longitude"]],
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
        popup="📍 Lookup Result",
        tooltip=f"📍 {lookup_result.get('City','?')}",
    ).add_to(m)

folium.LayerControl().add_to(m)

st_folium(m, width=None, height=650, returned_objects=[])


# ─────────────────────────────────────────────
# DATA TABLES & DOWNLOADS
# ─────────────────────────────────────────────
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🚁 Hub Coverage", "⚠️ High-Risk Areas", "📍 Hub Locations", "⭐ Recommended Hubs"]
)

with tab1:
    st.subheader("Hub Coverage Summary")
    st.dataframe(
        coverage[["HubID","PopulationCovered","ZIPsCovered",
                  "AvgDistanceMiles","AvgTravelMinutes","RiskExposure"]]
        .sort_values("RiskExposure", ascending=False)
        .style.format({
            "PopulationCovered": "{:,.0f}",
            "AvgDistanceMiles":  "{:.1f}",
            "AvgTravelMinutes":  "{:.0f}",
            "RiskExposure":      "{:,.0f}",
        }),
        use_container_width=True
    )
    st.download_button("⬇ Download CSV",
                       coverage.to_csv(index=False),
                       "hub_coverage.csv", "text/csv")

with tab2:
    st.subheader("Top 100 Highest-Risk ZIP Codes")
    cols = ["ZIP","City","State","Population","RiskWeight",
            "FloodRisk","HurricaneRisk","CoastalRisk",
            "NearestHub","DistanceMiles","TravelMinutes"]
    top100 = (df.sort_values("RiskWeight", ascending=False)
               .head(100)[cols])
    st.dataframe(
        top100.style.format({
            "Population":   "{:,.0f}",
            "RiskWeight":   "{:,.0f}",
            "FloodRisk":    "{:.3f}",
            "HurricaneRisk":"{:.3f}",
            "CoastalRisk":  "{:.3f}",
            "DistanceMiles":"{:.1f}",
            "TravelMinutes":"{:.0f}",
        }).background_gradient(subset=["RiskWeight"], cmap="Reds"),
        use_container_width=True
    )
    st.download_button("⬇ Download CSV",
                       top100.to_csv(index=False),
                       "high_risk_zips.csv", "text/csv")

with tab3:
    st.subheader("Optimized Hub Locations")
    st.dataframe(hubs, use_container_width=True)
    st.download_button("⬇ Download CSV",
                       hubs.to_csv(index=False),
                       "hubs.csv", "text/csv")

with tab4:
    st.subheader(f"Top {top_n} Recommended Hub Locations")
    st.dataframe(
        top_recommended[["City","State","ZIP","Latitude","Longitude",
                         "TotalScore","PopulationCovered"]]
        .style.format({
            "TotalScore":       "{:,.0f}",
            "PopulationCovered":"{:,.0f}",
        }),
        use_container_width=True
    )
    st.download_button("⬇ Download CSV",
                       top_recommended.to_csv(index=False),
                       "recommended_hubs.csv", "text/csv")
