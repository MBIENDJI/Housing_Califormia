import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import random

# ------------------------------
# Configuration de la page
# ------------------------------
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide"
)

# Fond orange clair avec design amélioré
st.markdown("""
<style>
    .stApp {
        background-color: #FFE5B4;
        background: linear-gradient(135deg, #FFE5B4 0%, #FFCC80 100%);
    }
    .main-header {
        text-align: center;
        color: #333333;
        padding: 20px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        margin-bottom: 30px;
        border-left: 8px solid #FF9800;
    }
    .step-container {
        background: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #FFD54F;
    }
    .step-title {
        color: #FF5722;
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .step-number {
        background: #FF9800;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }
    .prediction-card {
        background: linear-gradient(135deg, #4CAF50, #2E7D32);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        margin: 30px 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .stButton > button {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 152, 0, 0.4);
    }
    .info-box {
        background: #E3F2FD;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    .map-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stRadio > div {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
    }
    .stRadio > div > label {
        background: #FFF3E0;
        padding: 10px 20px;
        border-radius: 10px;
        border: 2px solid #FFB74D;
        transition: all 0.3s;
    }
    .stRadio > div > label:hover {
        background: #FFE0B2;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Charger le modèle
# ------------------------------
@st.cache_resource
def load_model():
    try:
        with open("xgboost_house_price_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'xgboost_house_price_model.pkl' not found!")
        st.stop()

model = load_model()

# ------------------------------
# DONNÉES RÉELLES de votre notebook - MODIFIÉES
# ------------------------------

# Traductions des catégories ocean_proximity
ocean_translations = {
    "ENGLISH": {
        "<1H OCEAN": "<1H OCEAN",
        "INLAND": "INLAND", 
        "NEAR BAY": "NEAR BAY",
        "NEAR OCEAN": "NEAR OCEAN"
    },
    "FRANCAIS": {
        "<1H OCEAN": "<1H OCÉAN",
        "INLAND": "INTÉRIEUR DES TERRES",
        "NEAR BAY": "PRÈS DE LA BAIE",
        "NEAR OCEAN": "PRÈS DE L'OCÉAN"
    },
    "ESPANA": {
        "<1H OCEAN": "<1H OCÉANO",
        "INLAND": "INTERIOR",
        "NEAR BAY": "CERCA DE LA BAHÍA",
        "NEAR OCEAN": "CERCA DEL OCÉANO"
    }
}

# IMPORTANT: Ces valeurs viennent de votre output (sans ISLAND)
ocean_encoded = {
    'NEAR BAY': 259230.8842289209,
    '<1H OCEAN': 240077.56560341993,
    'INLAND': 124789.5801620052,
    'NEAR OCEAN': 249365.78012048194
}

# Statistiques réelles par ocean_proximity (sans ISLAND)
ocean_stats = {
    '<1H OCEAN': {
        'lat': (32.61, 41.88),
        'lon': (-124.14, -116.62),
        'income': (4999.0, 150001.0),
        'pop_per_household': (1.07, 41.21),
        'age': (2, 52),
        'rooms_per_household': (0.85, 22.22),
        'count': 9123,
        'income_mean': 42309.57
    },
    'INLAND': {
        'lat': (32.64, 41.95),
        'lon': (-123.73, -114.31),
        'income': (4999.0, 150001.0),
        'pop_per_household': (1.06, 1243.33),
        'age': (1, 52),
        'rooms_per_household': (1.00, 141.91),
        'count': 6543,
        'income_mean': 32084.19
    },
    'NEAR BAY': {
        'lat': (37.35, 38.34),
        'lon': (-122.59, -122.01),
        'income': (4999.0, 150001.0),
        'pop_per_household': (1.28, 15.60),
        'age': (2, 52),
        'rooms_per_household': (1.55, 9.80),
        'count': 2289,
        'income_mean': 41738.80
    },
    'NEAR OCEAN': {
        'lat': (32.54, 41.95),
        'lon': (-124.35, -116.97),
        'income': (5360.0, 150001.0),
        'pop_per_household': (1.22, 502.46),
        'age': (2, 52),
        'rooms_per_household': (0.89, 28.60),
        'count': 2656,
        'income_mean': 40044.18
    }
}

# Points valides pour éviter l'eau (de votre output)
valid_points = {
    'NEAR BAY': [(37.8261, -122.2726), (38.0100, -122.2900), (37.8000, -122.2200)],
    '<1H OCEAN': [(34.5606, -118.8478), (34.0300, -118.0900), (37.3400, -118.2200)],
    'INLAND': [(35.5061, -118.5557), (36.7600, -118.0200), (34.1400, -119.8300)],
    'NEAR OCEAN': [(34.7394, -119.3336), (32.7100, -117.2200), (34.0400, -117.2300)]
}

# Plage de prédictions réelles
prediction_range = {
    'min': 23738.05,
    'max': 567930.94,
    'mean': 185299.06
}

# Valeurs globales moyennes
global_stats = {
    'rooms_per_household_mean': 5.4278,
    'income_mean': 38705.46,
    'population_per_household_mean': 3.0711
}

# ------------------------------
# Langues
# ------------------------------
languages = {
    "ENGLISH": {
        "welcome": "California Housing Price Predictor",
        "language_select": "Select your language",
        "choose_ocean": "1. Choose ocean proximity",
        "latitude": "2. Select latitude",
        "longitude": "3. Select longitude",
        "median_income": "4. Enter median income (USD)",
        "pop_per_household": "5. Population per household",
        "housing_age": "6. Housing median age (years)",
        "predict": "🔮 Predict House Value",
        "random_sample": "🎲 Generate Random Sample",
        "prediction_result": "Predicted House Value",
        "reset": "🔄 Reset to Default",
        "next": "Continue →",
        "back": "← Back",
        "step1_title": "🌎 Language Selection",
        "map_title": "📍 Selected Location in California",
        "income_info": "Enter value between ${:,.0f} and ${:,.0f}",
        "validation_error": "Please enter a value within the valid range",
        "predictions_range": "Model prediction range: ${:,.0f} - ${:,.0f}",
        "data_stats": "📊 Data Statistics",
        "samples_count": "Samples in category: {:,}",
        "ocean_categories": ocean_translations["ENGLISH"]
    },
    "FRANCAIS": {
        "welcome": "Prédicteur de Prix Immobilier en Californie",
        "language_select": "Choisissez votre langue",
        "choose_ocean": "1. Proximité de l'océan",
        "latitude": "2. Sélectionnez la latitude",
        "longitude": "3. Sélectionnez la longitude",
        "median_income": "4. Revenu médian (USD)",
        "pop_per_household": "5. Population par ménage",
        "housing_age": "6. Âge médian des logements (ans)",
        "predict": "🔮 Prédire la valeur",
        "random_sample": "🎲 Exemple aléatoire",
        "prediction_result": "Valeur prédite",
        "reset": "🔄 Réinitialiser",
        "next": "Continuer →",
        "back": "← Retour",
        "step1_title": "🌎 Sélection de la langue",
        "map_title": "📍 Emplacement sélectionné en Californie",
        "income_info": "Entrez une valeur entre ${:,.0f} et ${:,.0f}",
        "validation_error": "Veuillez entrer une valeur dans la plage valide",
        "predictions_range": "Plage de prédiction: ${:,.0f} - ${:,.0f}",
        "data_stats": "📊 Statistiques des données",
        "samples_count": "Échantillons dans la catégorie: {:,}",
        "ocean_categories": ocean_translations["FRANCAIS"]
    },
    "ESPANA": {
        "welcome": "Predictor de Precios de Vivienda en California",
        "language_select": "Seleccione su idioma",
        "choose_ocean": "1. Proximidad al océano",
        "latitude": "2. Seleccione latitud",
        "longitude": "3. Seleccione longitud",
        "median_income": "4. Ingreso medio (USD)",
        "pop_per_household": "5. Población por hogar",
        "housing_age": "6. Edad media de la vivienda (años)",
        "predict": "🔮 Predecir valor",
        "random_sample": "🎲 Muestra aleatoria",
        "prediction_result": "Valor predicho",
        "reset": "🔄 Restablecer",
        "next": "Continuar →",
        "back": "← Atrás",
        "step1_title": "🌎 Selección de idioma",
        "map_title": "📍 Ubicación seleccionada en Californie",
        "income_info": "Ingrese valor entre ${:,.0f} y ${:,.0f}",
        "validation_error": "Ingrese un valor dentro del rango válido",
        "predictions_range": "Rango de predicción: ${:,.0f} - ${:,.0f}",
        "data_stats": "📊 Estadísticas de datos",
        "samples_count": "Muestras en categoría: {:,}",
        "ocean_categories": ocean_translations["ESPANA"]
    }
}

# ------------------------------
# Initialisation de l'état de session
# ------------------------------
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'language' not in st.session_state:
    st.session_state.language = "ENGLISH"
if 'ocean_choice' not in st.session_state:
    st.session_state.ocean_choice = "<1H OCEAN"
if 'latitude' not in st.session_state:
    st.session_state.latitude = 34.56  # Moyenne pour <1H OCEAN
if 'longitude' not in st.session_state:
    st.session_state.longitude = -118.85  # Moyenne pour <1H OCEAN
if 'median_income' not in st.session_state:
    st.session_state.median_income = 42309.57  # Moyenne pour <1H OCEAN
if 'population_per_household' not in st.session_state:
    st.session_state.population_per_household = 3.05  # Moyenne pour <1H OCEAN
if 'housing_age' not in st.session_state:
    st.session_state.housing_age = 29  # Moyenne pour <1H OCEAN
if 'rooms_per_household' not in st.session_state:
    st.session_state.rooms_per_household = global_stats['rooms_per_household_mean']

# ------------------------------
# Étape 1 : Choix de la langue
# ------------------------------
if st.session_state.step == 1:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.title("🏠 California Housing Predictor")
        st.markdown("Predict house values using machine learning")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown('<div class="step-title"><div class="step-number">1</div>🌎 Language Selection</div>', unsafe_allow_html=True)
        
        lang_choice = st.selectbox(
            "Select your language / Choisissez votre langue / Seleccione su idioma",
            ["ENGLISH", "FRANCAIS", "ESPANA"],
            index=list(languages.keys()).index(st.session_state.language),
            key="lang_select"
        )
        
        if lang_choice != st.session_state.language:
            st.session_state.language = lang_choice
            st.rerun()
        
        t = languages[st.session_state.language]
        
        st.markdown("---")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("❌ Cancel", use_container_width=True):
                st.stop()
        with col_btn2:
            if st.button(f"✅ {t['next']}", use_container_width=True, type="primary"):
                st.session_state.step = 2
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Information sur le modèle
        with st.expander("ℹ️ About this predictor"):
            st.write("""
            **Model Information:**
            - Algorithm: XGBoost Regressor
            - Features used: 7 variables including location and demographics
            - Training samples: ~20,000 California housing records
            - Prediction range: ${:,.0f} - ${:,.0f}
            
            **Note:** Values are adjusted for inflation and represent median prices.
            """.format(prediction_range['min'], prediction_range['max']))

# ------------------------------
# Étape 2 : Interface principale
# ------------------------------
elif st.session_state.step == 2:
    t = languages[st.session_state.language]
    
    # Header principal
    st.markdown(f'<div class="main-header"><h1>🏠 {t["welcome"]}</h1></div>', unsafe_allow_html=True)
    
    # Bouton retour
    col_back, col_title, col_space = st.columns([1, 3, 1])
    with col_back:
        if st.button(f"← {t['back']}", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    
    # Layout principal
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        
        # Étape 1: Choix ocean_proximity (SEULEMENT 4 CATÉGORIES)
        st.markdown(f'<div class="step-title"><div class="step-number">1</div>{t["choose_ocean"]}</div>', unsafe_allow_html=True)
        
        # Les 4 catégories seulement
        ocean_options = list(ocean_stats.keys())  # ['<1H OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN']
        
        # Utiliser les traductions pour l'affichage
        ocean_display_names = [t['ocean_categories'][ocean] for ocean in ocean_options]
        
        # Créer un mapping pour retrouver la clé originale
        display_to_original = {t['ocean_categories'][ocean]: ocean for ocean in ocean_options}
        
        ocean_display_choice = st.radio(
            label=t["choose_ocean"],
            options=ocean_display_names,
            index=ocean_display_names.index(t['ocean_categories'][st.session_state.ocean_choice]) if st.session_state.ocean_choice in ocean_options else 0,
            key="ocean_radio"
        )
        
        # Convertir le choix d'affichage en clé originale
        ocean_choice = display_to_original[ocean_display_choice]
        
        if ocean_choice != st.session_state.ocean_choice:
            st.session_state.ocean_choice = ocean_choice
            # Réinitialiser avec les moyennes de la nouvelle catégorie
            stats = ocean_stats[ocean_choice]
            st.session_state.latitude = (stats['lat'][0] + stats['lat'][1]) / 2
            st.session_state.longitude = (stats['lon'][0] + stats['lon'][1]) / 2
            st.session_state.median_income = stats.get('income_mean', global_stats['income_mean'])
            st.session_state.population_per_household = stats['pop_per_household'][1] if stats['pop_per_household'][1] < 100 else 3.0
            st.session_state.housing_age = (stats['age'][0] + stats['age'][1]) // 2
            st.rerun()
        
        # Afficher la description
        st.caption(f"📊 {t['samples_count'].format(ocean_stats[ocean_choice]['count'])}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Étape 2: Latitude
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="step-title"><div class="step-number">2</div>{t["latitude"]}</div>', unsafe_allow_html=True)
        
        lat_range = ocean_stats[ocean_choice]['lat']
        latitude = st.slider(
            f"Latitude range: {lat_range[0]:.2f} to {lat_range[1]:.2f}",
            min_value=float(lat_range[0]),
            max_value=float(lat_range[1]),
            value=float(st.session_state.latitude),
            step=0.01,
            format="%.4f",
            key="latitude_slider"
        )
        st.session_state.latitude = latitude
        
        # Afficher un point valide suggéré
        if ocean_choice in valid_points and len(valid_points[ocean_choice]) > 0:
            valid_lat, valid_lon = valid_points[ocean_choice][0]
            st.caption(f"💡 Suggested valid point: {valid_lat:.4f}, {valid_points[ocean_choice][0][1]:.4f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Étape 3: Longitude
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="step-title"><div class="step-number">3</div>{t["longitude"]}</div>', unsafe_allow_html=True)
        
        lon_range = ocean_stats[ocean_choice]['lon']
        longitude = st.slider(
            f"Longitude range: {lon_range[0]:.2f} to {lon_range[1]:.2f}",
            min_value=float(lon_range[0]),
            max_value=float(lon_range[1]),
            value=float(st.session_state.longitude),
            step=0.01,
            format="%.4f",
            key="longitude_slider"
        )
        st.session_state.longitude = longitude
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Étape 4: Median Income
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="step-title"><div class="step-number">4</div>{t["median_income"]}</div>', unsafe_allow_html=True)
        
        income_range = ocean_stats[ocean_choice]['income']
        
        # Input avec validation
        median_income = st.number_input(
            t["income_info"].format(income_range[0], income_range[1]),
            min_value=float(income_range[0]),
            max_value=float(income_range[1]),
            value=float(st.session_state.median_income),
            step=1000.0,
            format="%.0f",
            key="income_input"
        )
        
        if not (income_range[0] <= median_income <= income_range[1]):
            st.error(t["validation_error"])
        
        st.session_state.median_income = median_income
        
        # Afficher la moyenne pour référence
        st.caption(f"📊 Mean for {t['ocean_categories'][ocean_choice]}: ${ocean_stats[ocean_choice].get('income_mean', income_range[0] + income_range[1]/2):,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Étape 5: Population per household
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="step-title"><div class="step-number">5</div>{t["pop_per_household"]}</div>', unsafe_allow_html=True)
        
        pop_range = ocean_stats[ocean_choice]['pop_per_household']
        # Limiter l'affichage si la valeur max est trop grande
        display_max = min(pop_range[1], 50) if pop_range[1] > 50 else pop_range[1]
        
        population_per_household = st.slider(
            f"Range: {pop_range[0]:.2f} to {pop_range[1]:.2f}",
            min_value=float(pop_range[0]),
            max_value=float(pop_range[1]),
            value=float(st.session_state.population_per_household),
            step=0.1,
            format="%.2f",
            key="pop_slider"
        )
        st.session_state.population_per_household = population_per_household
        
        if pop_range[1] > 50:
            st.info(f"⚠️ Maximum value is {pop_range[1]:.2f}, but typical values are below 10")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Étape 6: Housing age
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="step-title"><div class="step-number">6</div>{t["housing_age"]}</div>', unsafe_allow_html=True)
        
        age_range = ocean_stats[ocean_choice]['age']
        housing_age = st.slider(
            f"Age range: {age_range[0]} to {age_range[1]} years",
            min_value=int(age_range[0]),
            max_value=int(age_range[1]),
            value=int(st.session_state.housing_age),
            key="age_slider"
        )
        st.session_state.housing_age = housing_age
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Boutons d'action
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button(t["random_sample"], use_container_width=True):
                stats = ocean_stats[st.session_state.ocean_choice]
                st.session_state.latitude = random.uniform(stats['lat'][0], stats['lat'][1])
                st.session_state.longitude = random.uniform(stats['lon'][0], stats['lon'][1])
                st.session_state.median_income = random.uniform(stats['income'][0], stats['income'][1])
                
                # Pour population, éviter les valeurs extrêmes
                pop_max = min(stats['pop_per_household'][1], 10)
                st.session_state.population_per_household = random.uniform(stats['pop_per_household'][0], pop_max)
                
                st.session_state.housing_age = random.randint(stats['age'][0], stats['age'][1])
                st.success("✅ Random sample generated! Check the updated values above.")
                st.rerun()
        
        with col_act2:
            if st.button(t["reset"], use_container_width=True):
                stats = ocean_stats[st.session_state.ocean_choice]
                st.session_state.latitude = (stats['lat'][0] + stats['lat'][1]) / 2
                st.session_state.longitude = (stats['lon'][0] + stats['lon'][1]) / 2
                st.session_state.median_income = stats.get('income_mean', global_stats['income_mean'])
                st.session_state.population_per_household = min(3.0, stats['pop_per_household'][1])
                st.session_state.housing_age = (stats['age'][0] + stats['age'][1]) // 2
                st.success("✅ Reset to default values!")
                st.rerun()
        
        # Bouton de prédiction principal
        st.markdown("---")
        if st.button(t["predict"], type="primary", use_container_width=True):
            with st.spinner("🔄 Making prediction..."):
                try:
                    # Préparer les features dans le bon ordre
                    rooms_per_household = global_stats['rooms_per_household_mean']
                    
                    # IMPORTANT: Ordre exact comme dans l'entraînement
                    X_input = np.array([[
                        st.session_state.median_income,          # median_income
                        ocean_encoded[st.session_state.ocean_choice],  # ocean_proximity_encoded
                        rooms_per_household,                     # rooms_per_household
                        st.session_state.latitude,               # latitude
                        st.session_state.longitude,              # longitude
                        st.session_state.housing_age,            # housing_median_age
                        st.session_state.population_per_household # population_per_household
                    ]])
                    
                    # Faire la prédiction (le modèle retourne log(value))
                    pred_log = model.predict(X_input)[0]
                    
                    # Transformer inverse: exp(pred) - 1
                    pred_value = np.expm1(pred_log)
                    
                    # S'assurer que la prédiction est dans les limites raisonnables
                    pred_value = max(prediction_range['min'], min(prediction_range['max'], pred_value))
                    
                    # Afficher le résultat
                    st.markdown(f'''
                    <div class="prediction-card">
                        <h2>💰 {t["prediction_result"]}</h2>
                        <h1 style="font-size: 3em; margin: 20px 0;">${pred_value:,.0f}</h1>
                        <p style="font-size: 1.2em;">Median House Value Prediction</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Afficher les détails
                    with st.expander("📋 Prediction Details"):
                        st.write(f"**Input Features:**")
                        st.write(f"- Ocean Proximity: {t['ocean_categories'][st.session_state.ocean_choice]}")
                        st.write(f"- Latitude: {st.session_state.latitude:.4f}")
                        st.write(f"- Longitude: {st.session_state.longitude:.4f}")
                        st.write(f"- Median Income: ${st.session_state.median_income:,.0f}")
                        st.write(f"- Population/Household: {st.session_state.population_per_household:.2f}")
                        st.write(f"- Housing Age: {st.session_state.housing_age} years")
                        st.write(f"- Rooms/Household: {rooms_per_household:.2f}")
                        
                        st.write(f"\n**Model Output:**")
                        st.write(f"- Log Prediction: {pred_log:.4f}")
                        st.write(f"- Exponential Transform: {np.exp(pred_log):,.2f}")
                        st.write(f"- Final Prediction: ${pred_value:,.2f}")
                        
                        st.write(f"\n**Validation:**")
                        st.write(f"- Min Possible: ${prediction_range['min']:,.0f}")
                        st.write(f"- Max Possible: ${prediction_range['max']:,.0f}")
                        st.write(f"- Is in range: {prediction_range['min'] <= pred_value <= prediction_range['max']}")
                        
                except Exception as e:
                    st.error(f"❌ Prediction Error: {str(e)}")
                    st.info("""
                    **Troubleshooting:**
                    1. Verify model file exists
                    2. Check feature order matches training
                    3. Ensure all values are within valid ranges
                    """)
    
    with col_right:
        # Carte interactive
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="step-title">🗺️ {t["map_title"]}</div>', unsafe_allow_html=True)
        
        # Créer le DataFrame pour la carte avec le nom traduit
        df_map = pd.DataFrame({
            'lat': [st.session_state.latitude],
            'lon': [st.session_state.longitude],
            'size': [20],
            'category': [t['ocean_categories'][st.session_state.ocean_choice]],
            'label': ['Selected Location']
        })
        
        # Ajouter des points de référence
        reference_points = []
        if st.session_state.ocean_choice in valid_points:
            for i, (lat, lon) in enumerate(valid_points[st.session_state.ocean_choice]):
                reference_points.append({
                    'lat': lat,
                    'lon': lon,
                    'size': 10,
                    'category': 'Reference Point',
                    'label': f'Valid Point {i+1}'
                })
        
        if reference_points:
            df_map = pd.concat([df_map, pd.DataFrame(reference_points)], ignore_index=True)
        
        # Créer la carte
        fig = px.scatter_mapbox(
            df_map,
            lat='lat',
            lon='lon',
            color='category',
            size='size',
            hover_name='label',
            hover_data={'lat': ':.4f', 'lon': ':.4f', 'category': True},
            zoom=6,
            center={"lat": st.session_state.latitude, "lon": st.session_state.longitude},
            mapbox_style="open-street-map",
            height=500
        )
        
        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Statistiques de la catégorie
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="step-title">{t["data_stats"]}</div>', unsafe_allow_html=True)
        
        stats = ocean_stats[st.session_state.ocean_choice]
        ocean_display_name = t['ocean_categories'][st.session_state.ocean_choice]
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Category", ocean_display_name)
            st.metric("Samples", f"{stats['count']:,}")
            st.metric("Avg Latitude", f"{(stats['lat'][0] + stats['lat'][1])/2:.2f}")
        
        with col_stat2:
            st.metric("Avg Income", f"${stats.get('income_mean', global_stats['income_mean']):,.0f}")
            st.metric("Avg Longitude", f"{(stats['lon'][0] + stats['lon'][1])/2:.2f}")
            st.metric("Avg Age", f"{(stats['age'][0] + stats['age'][1])/2:.0f} yrs")
        
        st.caption(t["predictions_range"].format(prediction_range['min'], prediction_range['max']))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Information sur l'encodage
        with st.expander("🔧 Model Encoding Details"):
            st.write("**Ocean Proximity Encoding (Model Values):**")
            for ocean, code in ocean_encoded.items():
                display_name = t['ocean_categories'].get(ocean, ocean)
                st.write(f"- {display_name}: {code:,.2f}")
            
            current_display = t['ocean_categories'][st.session_state.ocean_choice]
            st.write(f"\n**Current Encoding:** {ocean_encoded.get(st.session_state.ocean_choice, 'N/A'):,.2f} ({current_display})")
            
            st.write("\n**Feature Order:**")
            st.code("""
['median_income',
 'ocean_proximity_encoded',
 'rooms_per_household',
 'latitude',
 'longitude',
 'housing_median_age',
 'population_per_household']
            """)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <strong>California Housing Price Predictor v2.0</strong> • 
        XGBoost Model • 
        <span style="color: #FF9800;">DataRockie Project</span><br/>
        <small style="color: #888;">
            Note: Predictions are estimates. Actual prices may vary based on market conditions.<br/>
            Model trained on California Housing Dataset with log-transformed target.
        </small>
    </div>
    """,
    unsafe_allow_html=True
)