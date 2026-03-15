HydroHub — Water Disaster Response Hub Optimizer
Optimizing emergency response hub placement across the U.S. to minimize travel time to communities at risk during floods, hurricanes, and coastal storms.
Built for the "Under the Sea" Hackathon.

Overview
Water-based disasters overwhelm emergency response systems and leave communities without help when they need it most. HydroHub uses real population and hazard data to answer one question: where should emergency hubs be placed so the most people can be reached in the least amount of time?

Features

Maps all U.S. ZIP codes with population and multi-hazard risk scores
Optimizes hub placement using weighted K-Means clustering
Assigns each ZIP to its nearest hub with distance and response time estimates
Interactive map with risk heatmaps, hub markers, and 200-mile coverage zones
Community lookup — enter any ZIP code or city name for an instant risk profile
Disaster scenario simulation with adjustable flood, hurricane, and coastal storm severity
Exports CSV reports for hub coverage, high-risk communities, and hub locations


Getting Started
Requirements
Python 3.9+
Install
bashgit clone https://github.com/yourusername/hydrohub.git
cd hydrohub
pip install -r requirements.txt
Run
bashstreamlit run app.py
Opens at http://localhost:8501.

Data
SourceUsed ForU.S. Census ZCTA GazetteerZIP code coordinatesFEMA National Risk IndexFlood, hurricane, and coastal risk scoresNOAA Storm EventsHistorical disaster damage estimatesU.S. Census Population DataPopulation by ZIP code
The app loads data in this order:

Local data/uszips.csv if present
U.S. Census ZCTA Gazetteer — downloads automatically, no login needed
Built-in synthetic dataset covering all 50 states as a fallback

The app always runs regardless of network availability.

How the Optimization Works
Weighted K-Means
ZIP codes are resampled proportional to population × risk score before K-Means runs. This pulls cluster centers toward high-risk, high-population areas instead of geographic midpoints.
Goal: Minimize Σ (Population × Risk × Distance to nearest hub)
Risk Scoring
Three scores per ZIP, derived from proximity to known hazard zones:

Flood Risk — Atlantic coast, Gulf Coast, inland flood corridors
Hurricane Risk — Gulf Coast and Atlantic seaboard exposure
Coastal Storm Risk — combined coastal proximity

Combined as:
RiskWeight = Population
           × (flood_weight × flood_severity × FloodRisk
           +  hurr_weight  × hurr_severity  × HurricaneRisk
           +  coastal_weight               × CoastalRisk)
           × (1 + HistoricalDamage / 1,000,000)
Travel Time
Vectorized haversine distance matrix — pairwise distances for 40,000+ ZIPs in under a second. Travel time estimated as (miles / 55 mph) × 60 + 15 min staging.

Project Structure
hydrohub/
├── app.py              # Streamlit application
├── requirements.txt
├── README.md
└── data/
    └── uszips.csv      # Auto-generated after first run

Dependencies
streamlit
pandas
numpy
scikit-learn
folium
streamlit-folium
requests
bashpip install -r requirements.txt

Controls
ControlWhat It DoesNumber of HubsSet how many hubs to place (3–60)Flood / Hurricane / Coastal WeightWeight each hazard type in the risk scoreFlood / Hurricane SeveritySimulate disaster intensity in real timeCommunity LookupSearch any ZIP or city for a full risk profileTop N Recommended HubsNumber of highest-scoring sites to highlight

Tabs and Exports
TabWhat You SeeHub Coverage & Response TimesEvery hub with city, population covered, avg travel timeHigh-Risk CommunitiesTop 100 ZIP codes by risk-weighted scoreOptimized Hub LocationsHub coordinates with nearest city and stateRecommended PlacementsHighest-impact candidate sites
Every tab has a Download CSV button.

What's Next

Real FEMA flood zone polygons for boundary-level risk accuracy
Road network routing via OpenStreetMap for realistic drive-time estimates
Animated flood and hurricane scenario overlays
Machine learning risk prediction trained on historical FEMA damage records
Live NOAA alert integration for real-time emergency planning


Built With
Python · Streamlit · Pandas · NumPy · scikit-learn · Folium · streamlit-folium · FEMA National Risk Index · NOAA Storm Events · U.S. Census ZCTA Gazetteer · Claude AI

License
MIT — free to use, modify, and build on.
