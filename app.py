import streamlit as st
import pandas as pd
import numpy as np
import os
import requests, zipfile, io, json
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from folium.plugins import MarkerCluster, HeatMap, TimestampedGeoJson
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape, Polygon
import geopandas as gpd
from geopy.geocoders import Nominatim
import networkx as nx
import osmnx as ox
st.set_page_config(layout="wide")

# -------------------
# CONFIG & PATHS
# -------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# API key from Streamlit secrets
ORS_API_KEY = st.secrets["ORS_API_KEY"]
@st.cache_data
def build_datasets():
    # -------------------
    # DOWNLOAD AND EXTRACT SIMPLEMAPS US ZIP CSV
    # -------------------
    zip_csv = os.path.join(DATA_DIR, "uszips.csv")
    if not os.path.exists(zip_csv):
        # SimpleMaps ZIP of CSV
        url = "https://simplemaps.com/static/data/us-zips/1.79/basic/simplemaps_uszips_basicv1.79.zip"
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(DATA_DIR)
        # Rename extracted CSV to uszips.csv for consistency
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                os.rename(os.path.join(DATA_DIR, name), zip_csv)
                break

    # Read CSV
    df_zip = pd.read_csv(zip_csv)

    # -------------------
    # Normalize columns
    # -------------------
    df_zip.columns = [c.lower().strip() for c in df_zip.columns]

    rename_map = {
        "zip": "ZIP",
        "zipcode": "ZIP",
        "zip_code": "ZIP",
        "postal_code": "ZIP",
        "lat": "Latitude",
        "latitude": "Latitude",
        "lng": "Longitude",
        "lon": "Longitude",
        "longitude": "Longitude",
        "population": "Population",
        "state_id": "State",
        "state": "State",
        "county": "County",
        "city": "City"
    }
    df_zip = df_zip.rename(columns={k:v for k,v in rename_map.items() if k in df_zip.columns})

    # -------------------
    # Ensure required columns exist
    # -------------------
    required_cols = ["ZIP", "Latitude", "Longitude"]
    for col in required_cols:
        if col not in df_zip.columns:
            st.error(f"ZIP dataset missing: {col}")
            st.write("Found columns:", df_zip.columns.tolist())
            st.stop()

    # Fill missing optional columns
    if "Population" not in df_zip.columns:
        df_zip["Population"] = np.random.randint(1000,50000,len(df_zip))
    if "County" not in df_zip.columns:
        df_zip["County"] = "Unknown"
    if "State" not in df_zip.columns:
        df_zip["State"] = "Unknown"

    # -------------------
    # FEMA flood/hurricane placeholder
    # -------------------
    fema_csv = os.path.join(DATA_DIR, "fema_risk.csv")
    if not os.path.exists(fema_csv):
        counties = df_zip[['County','State']].drop_duplicates()
        np.random.seed(0)
        counties['FloodRisk'] = np.random.uniform(0,1,len(counties))
        counties['HurricaneRisk'] = np.random.uniform(0,1,len(counties))
        counties.to_csv(fema_csv, index=False)

    df_fema = pd.read_csv(fema_csv)

    # Merge ZIPs with FEMA
    df = pd.merge(df_zip, df_fema, on=['County','State'], how='left')

    # Fill missing values
    df['FloodRisk'] = df['FloodRisk'].fillna(df['FloodRisk'].median())
    df['HurricaneRisk'] = df['HurricaneRisk'].fillna(df['HurricaneRisk'].median())
    df['HistoricalDamage'] = np.random.randint(5000,2000000,len(df))
    if 'Longitude' in df.columns:
        df['CoastalRisk'] = np.clip((df['Longitude'] > -90) * np.random.uniform(.2,.9,len(df)),0,1)
    else:
        df['CoastalRisk'] = np.random.uniform(.2,.9,len(df))

    return df
df = build_datasets()

# -------------------
# SIDEBAR CONTROLS
# -------------------
st.sidebar.title("HydroHub Controls")
hub_count = st.sidebar.slider("Number of Hubs", 3, 60, 15)
flood_mult = st.sidebar.slider("Flood Multiplier", 0.1, 3.0, 1.0)
hurr_mult = st.sidebar.slider("Hurricane Multiplier", 0.1, 3.0, 1.0)
coastal_mult = st.sidebar.slider("Coastal Storm Multiplier", 0.1, 3.0, 1.0)
zip_lookup = st.sidebar.text_input("ZIP Lookup")
top_n = st.sidebar.slider("Top N Recommended Hubs", 1, 20, 10)

# Scenario sliders
flood_scenario = st.sidebar.slider("Flood Severity Scale", 0.0, 2.0, 1.0)
hurricane_scenario = st.sidebar.slider("Hurricane Severity Scale", 0.0, 2.0, 1.0)

# -------------------
# COMPUTE WEIGHTED RISK
# -------------------
def compute_weight(df):
    df['RiskWeight'] = df['Population'] * (
        flood_mult*flood_scenario*df['FloodRisk'] + 
        hurr_mult*hurricane_scenario*df['HurricaneRisk'] + 
        coastal_mult*df['CoastalRisk']
    ) * (1 + df['HistoricalDamage']/1e6)
    return df

df = compute_weight(df)

# -------------------
# HUB OPTIMIZATION
# -------------------
def optimize_hubs(df, k):
    coords = df[['Latitude','Longitude']]
    model = KMeans(n_clusters=k, random_state=0, n_init=10)
    df['cluster'] = model.fit_predict(coords)
    hubs = pd.DataFrame(model.cluster_centers_, columns=['Latitude','Longitude'])
    hubs['HubID'] = range(len(hubs))
    return hubs

hubs = optimize_hubs(df, hub_count)

# -------------------
# REALISTIC TRAVEL TIMES (ORS API)
# -------------------
def compute_travel_times(df, hubs):
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    travel_times = []
    for i, row in df.iterrows():
        min_time = float('inf')
        for _, hub in hubs.iterrows():
            body = {"locations": [[row['Longitude'], row['Latitude']], [hub['Longitude'], hub['Latitude']]], "metrics":["duration"], "units":"m"}
            try:
                r = requests.post("https://api.openrouteservice.org/v2/matrix/driving-car", headers=headers, json=body)
                t = r.json()['durations'][0][1]/60
                if t < min_time:
                    min_time = t
            except:
                dist = np.sqrt((row['Latitude']-hub['Latitude'])**2 + (row['Longitude']-hub['Longitude'])**2)
                min_time = min(dist*111/60*1.4, min_time)
        travel_times.append(min_time)
    df['TravelMinutes'] = travel_times
    return df

df = compute_travel_times(df, hubs)

# -------------------
# ASSIGN NEAREST HUB
# -------------------
def assign_hubs(df, hubs):
    zip_coords = df[['Latitude','Longitude']]
    hub_coords = hubs[['Latitude','Longitude']]
    dist = cdist(zip_coords, hub_coords)
    df['NearestHub'] = dist.argmin(axis=1)
    df['Distance'] = dist.min(axis=1)
    return df

df = assign_hubs(df, hubs)

# -------------------
# COVERAGE METRICS
# -------------------
coverage = df.groupby("NearestHub").agg(
    PopulationCovered=("Population","sum"),
    AvgDistance=("Distance","mean"),
    AvgTravelMinutes=("TravelMinutes","mean"),
    RiskExposure=("RiskWeight","sum")
).reset_index()

# -------------------
# RECOMMENDED HUB LOCATIONS
# -------------------
df['HubScore'] = df['RiskWeight'] / (df['TravelMinutes'] + 1)
hub_candidates = df.groupby(['Latitude','Longitude']).agg(
    TotalScore=('HubScore','sum'),
    PopulationCovered=('Population','sum')
).reset_index()
top_recommended = hub_candidates.sort_values('TotalScore', ascending=False).head(top_n)

# -------------------
# DASHBOARD
# -------------------
st.title("HydroHub AI: Ultimate Emergency Hub Simulation")
st.write("Interactive, hackathon-ready dashboard for floods, hurricanes, and coastal storms.")

col1,col2,col3 = st.columns(3)
col1.metric("Population Modeled", int(df["Population"].sum()))
col2.metric("Average Travel Time (min)", round(df["TravelMinutes"].mean(),2))
col3.metric("High Risk ZIPs",(df["RiskWeight"]>df["RiskWeight"].quantile(.9)).sum())

# -------------------
# MAP
# -------------------
m = folium.Map(location=[39,-98], zoom_start=4)
cluster = MarkerCluster().add_to(m)

# Gradient risk points
sample = df.sample(min(5000,len(df)))
for _,row in sample.iterrows():
    color_intensity = int(min(255,row["RiskWeight"]/df["RiskWeight"].max()*255))
    color = f"#{color_intensity:02x}0000"
    folium.CircleMarker(
        location=[row["Latitude"],row["Longitude"]],
        radius=3,
        color=color,
        fill=True
    ).add_to(cluster)

# Add optimized hubs
for _,hub in hubs.iterrows():
    folium.Marker(
        [hub['Latitude'],hub['Longitude']],
        icon=folium.Icon(color='green'),
        popup=f"Hub {hub['HubID']}"
    ).add_to(m)

# Add recommended hubs
for _,row in top_recommended.iterrows():
    folium.Marker(
        [row['Latitude'], row['Longitude']],
        popup=f"Recommended Hub\nScore: {row['TotalScore']:.1f}",
        icon=folium.Icon(color='purple', icon='star')
    ).add_to(m)

# FEMA Flood Zones
try:
    fema_url = "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/0/query?where=1=1&outFields=*&f=geojson"
    r = requests.get(fema_url)
    flood_geo = r.json()
    folium.GeoJson(flood_geo, name="FEMA Flood Zones", style_function=lambda x: {"fillColor":"blue","color":"blue","weight":1,"fillOpacity":0.1}).add_to(m)
except:
    st.warning("Could not load FEMA flood zones.")

# ZIP Lookup
if zip_lookup:
    try:
        z = int(zip_lookup)
        r = df[df['ZIP']==z]
        if len(r)>0:
            r = r.iloc[0]
            folium.Marker(
                [r['Latitude'], r['Longitude']],
                icon=folium.Icon(color='red'),
                popup="Selected ZIP"
            ).add_to(m)
            st.subheader("ZIP Analysis")
            st.write("Nearest Hub:", r['NearestHub'])
            st.write("Distance:", round(r['Distance'],2))
            st.write("Travel Time (min):", round(r['TravelMinutes'],2))
            st.write("Flood Risk:", round(r['FloodRisk'],2))
            st.write("Hurricane Risk:", round(r['HurricaneRisk'],2))
            st.write("Coastal Risk:", round(r['CoastalRisk'],2))
        else:
            st.warning("ZIP not found")
    except:
        st.warning("Enter valid ZIP")

# Animated hurricane scenario
try:
    geojson_features = []
    for t in range(5):
        scaled_hurricane = df['HurricaneRisk']*hurricane_scenario*(t+1)/5
        for _, row in df.iterrows():
            geojson_features.append({
                "type":"Feature",
                "geometry":{"type":"Point","coordinates":[row['Longitude'],row['Latitude']]},
                "properties":{"time":f"2026-03-15T0{t}:00:00","style":{"color":"orange","radius":scaled_hurricane.iloc[_]*5,"fillColor":"orange"}}})
    TimestampedGeoJson({"type":"FeatureCollection","features":geojson_features}, period="PT1H", add_last_point=True).add_to(m)
except:
    st.warning("Could not load hurricane animation.")

# Risk heatmap
heat_data = [[row['Latitude'], row['Longitude'], row['RiskWeight']] for index,row in df.iterrows()]
HeatMap(heat_data, radius=15, blur=20, max_zoom=10).add_to(m)

# Render map
st_folium(m, width=1400, height=700)

# -------------------
# CSV DOWNLOADS
# -------------------
st.subheader("Hub Coverage")
st.dataframe(coverage)
st.download_button("Download Coverage CSV", coverage.to_csv(index=False), "hub_coverage.csv")

st.subheader("Top 50 High Risk Communities")
top = df.sort_values("RiskWeight", ascending=False).head(50)
st.dataframe(top[['ZIP','Population','RiskWeight','NearestHub','TravelMinutes']])
st.download_button("Download High Risk CSV", top.to_csv(index=False), "high_risk.csv")

st.subheader("Hub Locations")
st.dataframe(hubs)
st.download_button("Download Hubs CSV", hubs.to_csv(index=False), "hubs.csv")

st.subheader("Top Recommended Hub Locations")
st.dataframe(top_recommended[['Latitude','Longitude','TotalScore','PopulationCovered']])
st.download_button("Download Recommended Hubs CSV", top_recommended.to_csv(index=False), "recommended_hubs.csv")

