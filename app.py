import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from folium.plugins import MarkerCluster, HeatMap
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide")

# -------------------
# DATA PATH
# -------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------
# LOAD DATA
# -------------------
@st.cache_data
def load_data():
    # ZIP codes
    zip_file = os.path.join(DATA_DIR, "uszips.csv")
    df_zip = pd.read_csv(zip_file)
    df_zip = df_zip.rename(columns={"zip":"ZIP","lat":"Latitude","lng":"Longitude"})
    df_zip["Population"] = df_zip.get("population", pd.Series(np.random.randint(1000,50000,len(df_zip))))
    
    # FEMA risk
    fema_file = os.path.join(DATA_DIR, "fema_risk.csv")
    df_fema = pd.read_csv(fema_file)
    df_fema = df_fema.rename(columns={"COUNTY":"County","STATE":"State","FLOOD_RISK":"FloodRisk","STORM_RISK":"HurricaneRisk"})
    
    # Merge ZIP with FEMA risk
    df_zip['County'] = df_zip['county_name']
    df_zip['State'] = df_zip['state_id']
    df = pd.merge(df_zip, df_fema[['County','State','FloodRisk','HurricaneRisk']], on=['County','State'], how='left')
    
    # Fill missing risk values
    df['FloodRisk'] = df['FloodRisk'].fillna(df['FloodRisk'].median())
    df['HurricaneRisk'] = df['HurricaneRisk'].fillna(df['HurricaneRisk'].median())
    
    # Historical damage (placeholder)
    df['HistoricalDamage'] = np.random.randint(5000,2000000,len(df))
    
    # Coastal risk approximation
    df['CoastalRisk'] = np.clip((df['Longitude']>-90)*np.random.uniform(.2,.9,len(df)),0,1)
    
    return df

df = load_data()

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

# -------------------
# COMPUTE RISK WEIGHT
# -------------------
def compute_weight(df):
    df['RiskWeight'] = df['Population'] * (flood_mult*df['FloodRisk'] + hurr_mult*df['HurricaneRisk'] + coastal_mult*df['CoastalRisk']) * (1 + df['HistoricalDamage']/1e6)
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
# ASSIGN NEAREST HUB
# -------------------
def assign_hubs(df, hubs):
    zip_coords = df[['Latitude','Longitude']]
    hub_coords = hubs[['Latitude','Longitude']]
    dist = cdist(zip_coords, hub_coords)
    df['NearestHub'] = dist.argmin(axis=1)
    df['Distance'] = dist.min(axis=1)
    df['TravelMinutes'] = df['Distance'] * 111 / 60 * 1.4
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
st.title("HydroHub AI: Emergency Hub Optimization for Water Disasters")
st.write("Maximize response to floods, hurricanes, and coastal storms by optimizing hub placement.")

col1,col2,col3 = st.columns(3)
col1.metric("Population Modeled", int(df["Population"].sum()))
col2.metric("Average Travel Time (min)", round(df["TravelMinutes"].mean(),2))
col3.metric("High Risk ZIPs",(df["RiskWeight"]>df["RiskWeight"].quantile(.9)).sum())

# -------------------
# MAP
# -------------------
m = folium.Map(location=[39,-98], zoom_start=4)
cluster = MarkerCluster().add_to(m)

sample = df.sample(min(5000,len(df)))
for _,row in sample.iterrows():
    color="blue"
    if row["RiskWeight"]>df["RiskWeight"].quantile(.9):
        color="red"
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

# ZIP lookup
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

# Risk heatmap
st.subheader("Risk Heatmap + Hub Coverage")
heat_data = [[row['Latitude'], row['Longitude'], row['RiskWeight']] for index,row in df.iterrows()]
HeatMap(heat_data, radius=15, blur=20, max_zoom=10).add_to(m)

# Render map
st_folium(m, width=1400, height=700)

# Coverage table
st.subheader("Hub Coverage")
st.dataframe(coverage)
st.download_button("Download Coverage CSV", coverage.to_csv(index=False), "hub_coverage.csv")

# Top 50 high risk ZIPs
st.subheader("Top 50 High Risk Communities")
top = df.sort_values("RiskWeight", ascending=False).head(50)
st.dataframe(top[['ZIP','Population','RiskWeight','NearestHub','TravelMinutes']])
st.download_button("Download High Risk CSV", top.to_csv(index=False), "high_risk.csv")

# Hub locations
st.subheader("Hub Locations")
st.dataframe(hubs)
st.download_button("Download Hubs CSV", hubs.to_csv(index=False), "hubs.csv")

# Recommended hubs table
st.subheader("Top Recommended Hub Locations")
st.dataframe(top_recommended[['Latitude','Longitude','TotalScore','PopulationCovered']])
st.download_button("Download Recommended Hubs CSV", top_recommended.to_csv(index=False), "recommended_hubs.csv")
