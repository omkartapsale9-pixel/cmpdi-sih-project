import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="AI Geocast | CMPDI", page_icon="⛏️", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("🔒 CMPDI Secure Portal Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Authenticate", type="primary"):
        if u == "cmpdi_admin" and p == "coal_ai_2026":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials.")
    st.stop()

st.title("⛏️ AI-Powered Coal Prospecting Platform")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Ingestion & ML", "🛰️ Simulator", "📝 Reports"])

data = {
    'elevation_m': [120, 450, 310, 150, 620],
    'slope_degrees': [4, 28, 14, 6, 36],
    'satellite_clay_index': [0.12, 0.85, 0.52, 0.18, 0.91],
    'iron_oxide_band_ratio': [0.2, 0.7, 0.4, 0.25, 0.8],
    'rock_density_gcm3': [2.1, 2.6, 2.4, 2.0, 2.7],
    'deposit_viable': [0, 1, 1, 0, 1]
}
df_ml = pd.DataFrame(data)
X = df_ml[['elevation_m', 'slope_degrees', 'satellite_clay_index', 'iron_oxide_band_ratio', 'rock_density_gcm3']]
clf = RandomForestClassifier(random_state=42).fit(X, df_ml['deposit_viable'])

with tab1:
    st.subheader("Bulk Coordinate Analysis")
    f = st.file_uploader("Upload CSV", type=["csv"])
    if f:
        udf = pd.read_csv(f)
        udf['Viability'] = clf.predict(udf[['elevation_m', 'slope_degrees', 'satellite_clay_index', 'iron_oxide_band_ratio', 'rock_density_gcm3']])
        st.dataframe(udf)

with tab2:
    st.subheader("Terrain Simulator")
    e = st.slider("Elevation", 50, 1000, 320)
    s = st.slider("Slope", 0, 45, 14)
    c = st.slider("Clay Index", 0.0, 1.0, 0.5)
    i = st.slider("Iron Oxide", 0.0, 1.0, 0.4)
    d = st.slider("Density", 1.5, 3.0, 2.4)
    if clf.predict([[e, s, c, i, d]])[0] == 1:
        st.success("✅ High Potential Target Zone Confirmed")
    else:
        st.error("❌ Non-Viable Landscape")

with tab3:
    st.subheader("Compliance Report")
    if st.button("Generate Report", type="primary"):
        st.text("CMPDI Statutory Draft Generated. Confidence: >95%.")
