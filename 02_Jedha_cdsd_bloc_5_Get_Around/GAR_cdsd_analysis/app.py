import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
from pydantic import BaseModel

# --- Global Configuration and Data Loading ---

# Data/Model paths (must exist in your repository)
DELAY_ANALYSIS_URL = 'https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx'
PRICING_DATA_URL = "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv"

try:
    # NOTE: Ensure 'modele_GAR.joblib' is available in your Docker container or GitHub repo root.
    loaded_model = joblib.load('modele_GAR.joblib') 
    MODEL_LOADED = True
except Exception as e:
    # Error message if the model file is not found or fails to load
    st.error(f"Error loading model: {e}. Prediction feature is disabled. Please ensure 'modele_GAR.joblib' is in the root directory.")
    MODEL_LOADED = False


# Prediction Features Definition (used as reference for input fields)
class PredictionFeatures(BaseModel):
    model_key: str
    mileage: float
    engine_power: float
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

# --- CACHED DATA LOADING FUNCTIONS ---

@st.cache_data
def load_delay_data(url):
    """Loads the rental delay analysis dataset."""
    data_delay = pd.read_excel(url, sheet_name=['rentals_data', 'Documentation'])
    return data_delay['rentals_data']

# NOUVELLE FONCTION POUR PRÉPARER LES DONNÉES D'IMPACT DU RETARD
@st.cache_data
def calculate_delay_impact_data(data_delay_rentals):
    """
    Prépare les données pour l'analyse de l'impact du retard de la location précédente.
    """
    # on filtre les locations avec une location juste avant elles (previous_ended_rental_id > 0)
    condi2 = data_delay_rentals['previous_ended_rental_id'].notna()
    data_delay_rentals_data_withpreviousrental = data_delay_rentals[condi2].copy()

    # on trouve les valeurs de retard de checkout (delay_at_checkout_in_minutes) de la location précédente (previous_ended_rental_id)
    data_delay_rentals_data_withpreviousrental_withdelay = pd.merge(
        data_delay_rentals_data_withpreviousrental, 
        data_delay_rentals[['rental_id', 'delay_at_checkout_in_minutes']],
        left_on='previous_ended_rental_id',
        right_on='rental_id',
        how='left',
        suffixes=('_current', '_previous')
    )
    
    # Renommer et nettoyer
    data_delay_rentals_data_withpreviousrental_withdelay = data_delay_rentals_data_withpreviousrental_withdelay.rename(
        columns={'delay_at_checkout_in_minutes_previous': 'delay_at_checkout_in_minutes_from_previous_rental'}
    ).drop(columns=['rental_id_previous'])

    # on filtre les locations précédées d'une location avec un retard (delay_at_checkout_in_minutes_from_previous_rental > 0)
    # On utilise .notna() ici pour s'assurer que le merge a bien fonctionné et on garde seulement les retards positifs
    condi3 = (data_delay_rentals_data_withpreviousrental_withdelay['delay_at_checkout_in_minutes_from_previous_rental'].notna()) & \
             (data_delay_rentals_data_withpreviousrental_withdelay['delay_at_checkout_in_minutes_from_previous_rental'] > 0)
             
    data_filtered = data_delay_rentals_data_withpreviousrental_withdelay[condi3].copy()

    return data_filtered


@st.cache_data
def load_pricing_data(url):
    """Loads the pricing project dataset for the ML section."""
    df = pd.read_csv(url)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    return df

# --- APPLICATION PAGES ---

def rental_analysis_page():
    """Displays the rental delay impact analysis page."""
    
    st.title("📊 Impact of Time Delta & Previous Rental Delay")

    df_full = load_delay_data(DELAY_ANALYSIS_URL)
    
    # --- SECTION 1: Time Delta Analysis (Existing Code) ---
    
    st.header("1. Analysis of the Time Delta between consecutive rentals")
    
    total_rentals = len(df_full)
    
    # Calculate non-followed rentals
    df_notfollow = df_full[df_full['time_delta_with_previous_rental_in_minutes'].isna()]
    not_followed_count = len(df_notfollow)

    if total_rentals > 0:
        not_followed_percentage = round((not_followed_count / total_rentals) * 100, 2)
        st.info(
            f"Note: This analysis focuses only on rentals immediately followed by another. "
            f"**{not_followed_count} rentals ({not_followed_percentage}%)** were excluded from the primary delta analysis "
            f"because they were the last observed rental in a sequence (or had a missing time delta)."
        )

    # Display raw data checkbox
    if st.checkbox('Display raw data', False):
        st.subheader('Raw Data')
        st.dataframe(df_full)

    # --- 2. INTERACTIVE FILTER CREATION ---
    st.sidebar.subheader("Time Delta Filter")

    # Filter by Interval (Slider for a numerical column)
    df_clean = df_full.dropna(subset=['time_delta_with_previous_rental_in_minutes'])
    if df_clean.empty:
        st.error("Delay analysis data is empty or all missing.")
        return

    min_val = float(df_clean['time_delta_with_previous_rental_in_minutes'].min())
    max_val = float(df_clean['time_delta_with_previous_rental_in_minutes'].max())

    valeur_range = st.sidebar.slider(
        "Select minimum delta time to apply (in minutes):",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, min(max_val, 1440.0)), # Default range (max 24h)
        step=15.0
    )

    # --- 3. APPLYING THE FILTER ---
    
    # The filter applies to non-NaN data
    df_filtered_applied = df_clean.loc[
        (df_clean['time_delta_with_previous_rental_in_minutes'] >= valeur_range[0]) & 
        (df_clean['time_delta_with_previous_rental_in_minutes'] <= valeur_range[1])
    ]

    total_clean_rentals = len(df_clean)
    rentals_followed = len(df_filtered_applied)
    rentals_lost = total_clean_rentals - rentals_followed

    # --- 4. CREATION and DISPLAY of the Time Delta Histogram ---
    if not df_clean.empty:
        
        # 1. Calcul du pourcentage global de locations impactées
        global_impacted_pct = round((rentals_lost / total_rentals) * 100, 2)

        # 2. Calcul des locations perdues par type de check-in
        df_lost = df_clean.loc[
            (df_clean['time_delta_with_previous_rental_in_minutes'] < valeur_range[0]) | 
            (df_clean['time_delta_with_previous_rental_in_minutes'] > valeur_range[1])
        ]
        lost_by_checkin = df_lost.groupby('checkin_type').size().to_dict()

# --- Affichage des Métriques (Global) ---
        st.subheader("Metrics: Time Delta Impact")
        col_metrics_1, col_metrics_2, col_metrics_3, col_metrics_4 = st.columns(4)
        
        col_metrics_1.metric(
            label="Rentals with Follow-up (Known Delta)", 
            value=f"{total_clean_rentals}"
        )
        col_metrics_2.metric(
            label=f"Rentals Kept (Delta >= {int(valeur_range[0])} min)", 
            value=f"{rentals_followed}"
        )
        col_metrics_3.metric(
            label=f"Potentially Impacted Rentals", 
            value=f"{rentals_lost}",
            delta=f"{-rentals_lost}", 
            delta_color="normal"
        )
        col_metrics_4.metric(
            label="Global Rentals Impacted %",
            value=f"{global_impacted_pct}%"
        )


        st.subheader("Distribution of Time Delta Between Rentals (Filtered)")
        
        if not df_filtered_applied.empty:
            # Creating the histogram with Plotly Express
            fig = px.histogram(
                df_filtered_applied, 
                x="time_delta_with_previous_rental_in_minutes",
                color='checkin_type', 
                barmode='overlay',
                #nbins=int((valeur_range[1] - valeur_range[0]) / 15) if (valeur_range[1] - valeur_range[0]) > 0 else 1,
                title=f"Distribution of Rentals between {int(valeur_range[0])} and {int(valeur_range[1])} minutes delta"
            )

            fig.update_layout(
                xaxis_title="Time Delta Between Rentals (minutes)",
                yaxis_title="Frequency (Number of Observations)",
                hovermode="x unified",
                xaxis=dict(
                    range=[0, 720],
                )
            )
            
            # Displaying the Plotly figure in Streamlit
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No data matches the selected filters. Please adjust the criteria.")

    else:
        st.warning("⚠️ No data could be loaded for delta time analysis.")
    
    
    st.markdown("---")
    st.header("2. Impact of the Previous Rental's Checkout Delay")
    
    df_delay_impact = calculate_delay_impact_data(df_full)

    if df_delay_impact.empty:
        st.warning("⚠️ No data available for delay impact analysis (no rentals with a delayed previous rental).")
        return

    st.sidebar.subheader("Previous Delay Filter")
    
    max_delay = float(df_delay_impact['delay_at_checkout_in_minutes_from_previous_rental'].max())
    
    default_max_view = min(max_delay, 2000.0) 

    delay_range = st.sidebar.slider(
        "Select max delay from previous rental to display (minutes):",
        min_value=0.0,
        max_value=max_delay,
        value=default_max_view, 
        step=10.0
    )
    
    df_viz = df_delay_impact[
        df_delay_impact['delay_at_checkout_in_minutes_from_previous_rental'] <= delay_range
    ].copy()


    if not df_viz.empty:
        
        total_impacted = len(df_delay_impact)
        kept_for_viz = len(df_viz)
        
        impact_pct_of_total_clean = round((total_impacted / total_rentals) * 100, 2)


        col_imp_1, col_imp_2, col_imp_3 = st.columns(3)
        
        col_imp_1.metric(
            label="Total Rentals Impacted by Previous Delay (P.D. > 0)", 
            value=f"{total_impacted}"
        )
        
        col_imp_2.metric(
            label="Percentage of Rentals Impacted by the Delay of the Previous Rental",
            value=f"{impact_pct_of_total_clean}%",
        )
        
        col_imp_3.metric(
            label=f"Rentals Displayed (P.D. <= {int(delay_range)} min)", 
            value=f"{kept_for_viz}"
        )

        st.subheader("Distribution of the Previous Rental's Delay by Current Rental State")

        fig_delay_impact = px.histogram(
            df_viz, 
            x="delay_at_checkout_in_minutes_from_previous_rental",
            color='state',
            barmode='overlay',
            nbins=50,
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title="Impact of the Delay on the State of Rental"
        )
        
        fig_delay_impact.update_traces(opacity=0.6, marker_line_color='black', marker_line_width=1)
        
        fig_delay_impact.update_layout(
            xaxis_title="Delay of the Previous Rental at Checkout (minutes)",
            yaxis_title="Frequency (Number of Observations)",
            hovermode="x unified",
            xaxis=dict(range=[0, delay_range]) 
        )

        st.plotly_chart(fig_delay_impact, use_container_width=True)
    else:
        st.warning("⚠️ No data to display for the selected maximum delay. Please increase the limit.")


def prediction_page():
    """Displays the price prediction page and includes an option to display raw data."""
    st.title("💰 Daily Rental Price Prediction")

    if not MODEL_LOADED:
        st.error("Machine Learning model could not be loaded. Feature disabled.")
        return
    
    pricing_df = load_pricing_data(PRICING_DATA_URL)
    if st.checkbox('Display raw pricing data', False):
        st.subheader('Raw Pricing Data Preview')
        st.dataframe(pricing_df.head(100)) 
        st.caption(f"Total rows in dataset: {len(pricing_df)}. Showing first 100 rows.")
    
    st.markdown("---")
    st.subheader("Enter vehicle features:")

    # Using Streamlit columns for clean layout
    col1, col2, col3 = st.columns(3)

    # --- Column 1: Model and Power ---
    with col1:
        model_key = st.selectbox("Model (model_key)", ['Peugeot', 'Renault', 'BMW', 'Audi', 'Ferrari'])
        engine_power = st.slider("Engine Power (hp)", min_value=50.0, max_value=400.0, value=135.0, step=1.0)
        mileage = st.number_input("Mileage (mileage)", min_value=0.0, value=100000.0, step=1000.0)

    # --- Column 2: Color and Type ---
    with col2:
        fuel = st.selectbox("Fuel (fuel)", ['diesel', 'petrol', 'hybrid', 'electric'])
        paint_color = st.selectbox("Color (paint_color)", ['black', 'grey', 'white', 'red', 'blue', 'other'])
        car_type = st.selectbox("Car Type (car_type)", ['sedan', 'coupe', 'hatchback', 'suv', 'stationwagon'])

    # --- Column 3: Features ---
    with col3:
        st.markdown("### Features (bool)")
        private_parking_available = st.checkbox("Private Parking Available", value=True)
        has_gps = st.checkbox("GPS", value=True)
        has_air_conditioning = st.checkbox("Air Conditioning", value=False)
        automatic_car = st.checkbox("Automatic Car", value=False)
        has_getaround_connect = st.checkbox("GetAround Connect", value=True)
        has_speed_regulator = st.checkbox("Speed Regulator", value=False)
        winter_tires = st.checkbox("Winter Tires", value=True)


    # --- Prediction Button ---
    st.markdown("---")
    if st.button("Calculate Estimated Rental Price", type="primary"):
        # 1. Prepare data in the format expected by the model
        input_data = pd.DataFrame({
            "model_key": [model_key],
            "mileage": [mileage],
            "engine_power": [engine_power],
            "fuel": [fuel],
            "paint_color": [paint_color],
            "car_type": [car_type],
            "private_parking_available": [private_parking_available],
            "has_gps": [has_gps],
            "has_air_conditioning": [has_air_conditioning],
            "automatic_car": [automatic_car],
            "has_getaround_connect": [has_getaround_connect],
            "has_speed_regulator": [has_speed_regulator],
            "winter_tires": [winter_tires]
        })

        # 2. Prediction (replaces API call)
        try:
            prediction = loaded_model.predict(input_data)
            predicted_price = round(prediction.tolist()[0], 2)
            
            st.success(f"## ✅ Prediction Successful")
            st.balloons()
            st.metric(label="Estimated Rental Price (per day)", value=f"{predicted_price} €")
            st.caption("This estimate is based on the loaded Machine Learning model.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.warning("Please check the model input types.")


# --- MAIN FUNCTION FOR PAGE MANAGEMENT ---
def main():
    st.set_page_config(
        page_title="GetAround Insights",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.sidebar.title("Navigation")
    
    # Page selection menu in the sidebar
    page = st.sidebar.radio("Select Feature", 
                            ["Delay Analysis", "ML Price Prediction"])

    if page == "Delay Analysis":
        rental_analysis_page()
    elif page == "ML Price Prediction":
        prediction_page()

if __name__ == "__main__":
    main()