import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import base64  # For encoding images
import numpy as np # For map data
import pydeck as pdk # Import for the map


# PAGE CONFIG
# This MUST be the first Streamlit command in your app
st.set_page_config(
    page_title="SustainOps",
    page_icon="🌿",
    layout="wide",
)

# HELPER FUNCTION
@st.cache_data
def get_img_as_base64(file):
    """Reads a local image file and returns it as a Base64 encoded string."""
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Image file not found: {file}. Please make sure it's in the correct folder.")
        return None


# DATA LOADING
@st.cache_data
def load_data():
    """Loads and processes the main dataset."""
    try:
        data = pd.read_csv("electricityConsumptionAndProductioction.csv")
    except FileNotFoundError:
        return None, None # Return None if file not found

    # Convert DateTime column to datetime objects
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    
    # --- SIMULATE CARBON EMISSIONS ---
    emission_factors = {
        'Coal': 0.82, 'Oil and Gas': 0.49, 'Nuclear': 0.012,
        'Wind': 0.011, 'Hydroelectric': 0.024, 'Biomass': 0.23,
        'Solar': 0.045
    }

    emissions_cols = []
    for source, factor in emission_factors.items():
        if source in data.columns:
            data[f'{source}_emissions'] = data[source] * factor
            emissions_cols.append(f'{source}_emissions')
            
    data['Total Carbon Emissions (tons)'] = data[emissions_cols].sum(axis=1)
    data.set_index('DateTime', inplace=True)
    return data, emissions_cols


# LOGIN PAGE
def show_login_page():
    """Displays the split-screen login page with a high-visibility Continue button."""
    
    # 1. Get the background image
    img = get_img_as_base64("Login_bg.jpg") 
    if img is None:
        st.error("Login background image 'Login_bg.jpg' not found. App cannot start.")
        st.stop()

    # 2. Define the CSS for the split-screen layout
    login_page_style = f"""
    <style>
    /* --- Base App Overrides --- */
    [data-testid="stAppViewContainer"] > .main {{
        padding: 0 !important;
        margin: 0 !important;
        background: #0E1117;
    }}
    [data-testid="stHorizontalBlock"] {{
        gap: 0 !important;
    }}

    /* --- Left Column (Image) --- */
    [data-testid="stHorizontalBlock"] > div:nth-child(1) {{
        background-image: url("data:image/png;base64,{img}") !important;
        background-size: cover !important;
        background-position: center !important;
        min-height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        padding: 2rem 4rem !important;
    }}
    
    /* --- Right Column (Form) --- */
    [data-testid="stHorizontalBlock"] > div:nth-child(2) {{
        background-color: white !important;
        min-height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        padding: 3rem 5rem !important;
    }}
    
    /* --- Force text to be dark in right column --- */
    [data-testid="stHorizontalBlock"] > div:nth-child(2) label,
    [data-testid="stHorizontalBlock"] > div:nth-child(2) p,
    [data-testid="stHorizontalBlock"] > div:nth-child(2) h1 {{
        color: #111111 !important;
    }}

    /* --- HIGH VISIBILITY CONTINUE BUTTON --- */
    /* Targeting the button inside the Streamlit Form */
    div[data-testid="stFormSubmitButton"] > button {{
        background-color: #000000 !important; /* Pure Black */
        color: #ffffff !important; /* Pure White Text */
        width: 100% !important;
        height: 3.5rem !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 800 !important; /* Extra Bold */
        font-size: 1.2rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }}
    
    div[data-testid="stFormSubmitButton"] > button:hover {{
        background-color: #333333 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3) !important;
    }}

    /* --- Input Field Improvements --- */
    [data-testid="stHorizontalBlock"] > div:nth-child(2) input {{
        border: 1px solid #ced4da !important;
        border-radius: 8px !important;
    }}
    </style>
    """
    st.markdown(login_page_style, unsafe_allow_html=True)

    # 3. Create the two columns
    col1, col2 = st.columns([1.2, 1], gap="small") 

    # --- Left Column Content ---
    with col1:
        st.title("Welcome to SustainOps!")
        st.subheader("The AI-powered platform for intelligent carbon emission management.")

    # --- Right Column Content ---
    with col2:
        st.image("Logo.jpg", width=60) 
        st.title("Sign in")
        st.write("Enter your credentials to access the system")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="admin") 
            password = st.text_input("Password", type="password", placeholder="demo123")
            
            st.markdown("""
                <div style='text-align: right; margin-top: -10px; margin-bottom: 10px;'>
                    <a href='#' target='_blank' style='color: #007bff; text-decoration: none; font-size: 0.9rem;'>
                        Forgot Password?
                    </a>
                </div>
            """, unsafe_allow_html=True)
            
            submitted = st.form_submit_button("Continue")
            
            if submitted:
                if username == "admin" and password == "demo123": 
                    st.session_state['is_logged_in'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        st.divider()
        st.button("Sign in with Google", use_container_width=True, disabled=True, help="Single Sign-On (SSO) not yet enabled")



# PAGE 1: OVERVIEW (MONITOR)
def page_overview(data, emissions_cols):
    st.title("🌿 SustainOps")
    st.subheader("Intelligent Carbon Emission Dashboard")
    
    monthly_data = data.resample('ME').sum()
    
    # --- ROW 1: KPIs ---
    st.markdown("### Key Metrics (Last Month)")
    
    last_month_data = monthly_data.iloc[-2]
    prev_month_data = monthly_data.iloc[-3]
    
    total_emissions = last_month_data['Total Carbon Emissions (tons)']
    emissions_delta = ((total_emissions - prev_month_data['Total Carbon Emissions (tons)']) / prev_month_data['Total Carbon Emissions (tons)']) * 100
    
    total_consumption = last_month_data['Consumption']
    consumption_delta = ((total_consumption - prev_month_data['Consumption']) / prev_month_data['Consumption']) * 100
    
    last_month_emissions = last_month_data[emissions_cols]
    top_source = last_month_emissions.idxmax().replace('_emissions', '')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Carbon Emissions", value=f"{total_emissions:,.0f} tons", delta=f"{emissions_delta:.2f}% vs. previous month")
    with col2:
        st.metric(label="Total Energy Consumption", value=f"{total_consumption:,.0f} MWh", delta=f"{consumption_delta:.2f}% vs. previous month")
    with col3:
        st.metric(label="Top Emitting Source", value=top_source)
    
    st.divider()

    # --- ROW 2: HISTORICAL FOOTPRINT ---
    st.markdown("### Historical Carbon Footprint (Monthly)")
    fig_hist = px.line(monthly_data, y='Total Carbon Emissions (tons)', title='Total Carbon Emissions Over Time')
    fig_hist.update_layout(xaxis_title='Date', yaxis_title='Carbon Emissions (tons)')
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # --- ROW 3: EMISSIONS BREAKDOWN & NEW MAP ---
    st.markdown("### Emissions Breakdown (Last 12 Months)")
    col1, col2 = st.columns([1, 1.5]) # Give map more space
    
    with col1:
        last_12_months = monthly_data.iloc[-12:]
        emissions_summary = last_12_months[emissions_cols].sum()
        emissions_summary.index = emissions_summary.index.str.replace('_emissions', '')
        
        fig_donut = px.pie(
            emissions_summary, 
            values=emissions_summary.values, 
            names=emissions_summary.index,
            title="Emissions by Source",
            hole=0.4 # This creates the donut hole
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with col2:
        st.markdown("### Emissions by Facility (Mock Data)")
        
        map_data = pd.DataFrame({
            "facility": ["Berlin HQ", "Madrid Factory", "London Office", "Paris R&D"],
            "lat": [52.5200, 40.4168, 51.5074, 48.8566],
            "lon": [13.4050, -3.7038, -0.1278, 2.3522],
            "emissions": [120000, 450000, 55000, 90000],
        })
        
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=48.8,
                longitude=2.35,
                zoom=4,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                   'HexagonLayer',
                   data=map_data,
                   get_position='[lon, lat]',
                   radius=100000,
                   elevation_scale=100,
                   elevation_range=[0, 1000],
                   pickable=True,
                   extruded=True,
                ),
                pdk.Layer(
                    'ScatterplotLayer',
                    data=map_data,
                    get_position='[lon, lat]',
                    get_color='[200, 30, 0, 160]',
                    get_radius=100000,
                ),
            ],
            tooltip={"text": "{facility}\nEmissions: {emissions} tons"}
        ))

# -----------------
# PAGE 2: FORECASTING (ANALYZE)
# -----------------
@st.cache_resource # Cache the model
def train_forecast_model(df_prophet):
    m = Prophet(seasonality_mode='multiplicative', daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df_prophet)
    return m

def page_forecasting(data, emissions_cols):
    st.title("📈 Forecasting & Analysis")
    st.subheader("Predictive Modeling for Carbon Emissions")
    st.markdown("### Forecasting Controls")
    
    df_daily = data['Total Carbon Emissions (tons)'].resample('D').sum()
    df_prophet = df_daily.reset_index()
    df_prophet.columns = ['ds', 'y']

    horizon_options = {"30 Days": 30, "3 Months": 90, "6 Months": 182, "1 Year": 365}
    horizon_label = st.selectbox("Select Forecast Horizon", horizon_options.keys())
    periods_to_forecast = horizon_options[horizon_label]

    if st.button("Run Forecast"):
        with st.spinner(f"Running forecast for {horizon_label}..."):
            
            m = train_forecast_model(df_prophet)
            
            future = m.make_future_dataframe(periods=periods_to_forecast)
            forecast = m.predict(future)

            st.markdown("### Future Emissions Forecast")
            fig_forecast = plot_plotly(m, forecast)
            fig_forecast.update_layout(xaxis_title='Date', yaxis_title='Carbon Emissions (tons)')
            st.plotly_chart(fig_forecast, use_container_width=True)

            st.markdown("### Forecast Components")
            fig_components = m.plot_components(forecast)
            st.pyplot(fig_components)
            
            # Store last forecast value for "What-If"
            st.session_state['last_forecast_value'] = forecast['yhat'].iloc[-1]
            st.session_state['last_forecast_period'] = horizon_label

    # --- NEW: What-If Scenario Analysis ---
    if 'last_forecast_value' in st.session_state:
        st.divider()
        st.markdown("### 🎯 What-If Scenario Analysis")
        st.write(f"Based on your **{st.session_state['last_forecast_period']}** forecast:")
        
        baseline_emissions = st.session_state['last_forecast_value']
        
        col1, col2 = st.columns(2)
        with col1:
            # Get top sources
            monthly_data = data.resample('ME').sum()
            last_12_months = monthly_data.iloc[-12:]
            emissions_summary = last_12_months[emissions_cols].sum()
            top_source = emissions_summary.idxmax().replace('_emissions', '')
            second_source = emissions_summary.drop(emissions_summary.idxmax()).idxmax().replace('_emissions', '')

            # Simple percentage contribution (mock)
            top_source_contrib = 0.4 # Assuming 40%
            second_source_contrib = 0.3 # Assuming 30%
            
            reduce_top = st.slider(f"Reduce {top_source} usage by:", 0, 50, 10, format="%d%%")
            reduce_second = st.slider(f"Reduce {second_source} usage by:", 0, 50, 5, format="%d%%")
            
            # Calculate simulated reduction
            reduction_top = baseline_emissions * top_source_contrib * (reduce_top / 100)
            reduction_second = baseline_emissions * second_source_contrib * (reduce_second / 100)
            total_reduction = reduction_top + reduction_second
            new_forecasted_emissions = baseline_emissions - total_reduction

        with col2:
            st.markdown("#### Simulated Impact:")
            st.metric(
                label=f"New Forecasted Emissions ({st.session_state['last_forecast_period']})",
                value=f"{new_forecasted_emissions:,.0f} tons",
                delta=f"-{total_reduction:,.0f} tons (Saved)"
            )
            st.info(f"This simulation assumes reducing **{top_source}** by {reduce_top}% and **{second_source}** by {reduce_second}% will directly reduce your total emissions.")

# -----------------
# PAGE 3: RECOMMENDATIONS (OPTIMIZE)
# -----------------
def initialize_recommendations():
    """Initializes the recommendations DataFrame in session_state."""
    if 'recommendations_df' not in st.session_state:
        st.session_state.recommendations_df = pd.DataFrame([
            {"Action": "Review Energy Source", "Priority": "🔴 High", "Est. Impact (tCO2/yr)": -25000, "Status": "To Do", "Complete": False},
            {"Action": "Investigate Recent Spike", "Priority": "🔴 High", "Est. Impact (tCO2/yr)": -10000, "Status": "To Do", "Complete": False},
            {"Action": "Improve Operational Efficiency (HVAC)", "Priority": "🟠 Medium", "Est. Impact (tCO2/yr)": -8000, "Status": "To Do", "Complete": False},
            {"Action": "Implement 'Power-Down' Schedule", "Priority": "🟠 Medium", "Est. Impact (tCO2/yr)": -4500, "Status": "To Do", "Complete": False},
            {"Action": "Enhance Data Granularity (Sub-metering)", "Priority": "🟢 Low", "Est. Impact (tCO2/yr)": 0, "Status": "To Do", "Complete": False},
        ])

def page_recommendations(data, emissions_cols):
    st.title("💡 Recommendations & Optimization")
    st.subheader("Actionable Insights to Reduce Your Carbon Footprint")
    
    initialize_recommendations()

    st.markdown("### Total Savings Tracker")
    
    # Calculate savings from completed actions
    df = st.session_state.recommendations_df
    completed_df = df[df["Complete"] == True]
    total_savings = completed_df["Est. Impact (tCO2/yr)"].sum()
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.metric(
            label="Total CO2 Saved from Completed Actions",
            value=f"{-total_savings:,.0f} tons / year"
        )
        st.write("Check 'Complete' in the table below to update this tracker.")
    
    with col2:
        # New chart
        st.markdown("#### Potential Impact of All Actions")
        # Filter out 0-impact actions for a cleaner chart
        impact_df = df[df["Est. Impact (tCO2/yr)"] < 0].copy()
        impact_df["Est. Impact (tCO2/yr)"] = impact_df["Est. Impact (tCO2/yr)"] * -1 # Make positive for chart
        
        fig_impact = px.bar(
            impact_df,
            x="Est. Impact (tCO2/yr)",
            y="Action",
            orientation='h',
            title="Estimated CO2 Savings per Action"
        )
        fig_impact.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_impact, use_container_width=True)
    
    st.divider()

    st.markdown("### Prioritized Action List")
    st.write("Track your progress on AI-powered recommendations. Edit the status below.")

    edited_df = st.data_editor(
        st.session_state.recommendations_df,
        column_config={
            "Action": st.column_config.TextColumn(width="large"),
            "Priority": st.column_config.SelectboxColumn(
                options=["🔴 High", "🟠 Medium", "🟢 Low"],
                width="small"
            ),
            "Status": st.column_config.SelectboxColumn(
                options=["To Do", "In Progress", "Done", "On Hold"],
                width="medium"
            ),
            "Complete": st.column_config.CheckboxColumn(
                width="small"
            ),
            "Est. Impact (tCO2/yr)": st.column_config.NumberColumn(
                format="%d"
            )
        },
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic" # Allows adding/deleting rows
    )
    
    # Update the session state with the edited dataframe
    st.session_state.recommendations_df = edited_df


# MAIN DASHBOARD FUNCTION
def show_dashboard():
    """Shows the main dashboard after login."""
    
    # Load data
    data, emissions_cols = load_data()
    
    if data is None or emissions_cols is None:
        st.error("Fatal Error: Could not load data file. Please ensure 'electricityConsumptionAndProductioction.csv' is in the same folder as app.py.")
        st.stop() # Stop the app if data is missing

    # --- NAVIGATION SIDEBAR ---
    st.sidebar.title(f"Welcome, {st.session_state['username']}!")
    
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state['is_logged_in'] = False
        st.session_state.pop('username', None) # Remove username on logout
        st.rerun()

    st.sidebar.divider()
    st.sidebar.title("Navigation")
    page_options = ["🏠 Overview (Monitor)", "📈 Forecasting (Analyze)", "💡 Recommendations (Optimize)"]
    page = st.sidebar.radio("Go to", page_options)

    # --- PAGE ROUTING ---
    if page == "🏠 Overview (Monitor)":
        page_overview(data, emissions_cols)
    elif page == "📈 Forecasting (Analyze)":
        page_forecasting(data, emissions_cols) 
    elif page == "💡 Recommendations (Optimize)":
        page_recommendations(data, emissions_cols)


# MAIN APP ROUTER
def main():
    """Main function to route between login and dashboard."""
    if 'is_logged_in' not in st.session_state:
        st.session_state['is_logged_in'] = False
        
    if st.session_state['is_logged_in']:
        show_dashboard()
    else:
        show_login_page()

if __name__ == "__main__":
    main()