import streamlit as st 
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder


# ============================
# Streamlit config
# ============================
st.set_page_config(page_title="Prédiction Retards Vols", page_icon="✈️", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.75), rgba(255, 255, 255, 0.75)),
                    url('https://www.discoverpuertorico.com/sites/default/files/styles/share_image/public/2022-12/Airplane%20at%20Luis%20Muoz%20Marn%20International%20Airport.jpg?itok=2Irhb0Bf');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Système de Prédiction des Retards de Vols")


# ============================
# Chargement dataset (uniques)
# ============================

@st.cache_data
def load_unique_values():
    df = pd.read_csv("cleaned_flights.csv")
    unique = {
        "origin": sorted(df["origin"].dropna().unique()),
        "dest": sorted(df["dest"].dropna().unique()),
        "uniquecarrier": sorted(df["uniquecarrier"].dropna().unique()),
        "tailnum": sorted(df["tailnum"].dropna().unique()),
        "origincityname": sorted(df["origincityname"].dropna().unique()),
    }
    return unique

unique_vals = load_unique_values()


# ============================
# Chargement dataset complet
# ============================

@st.cache_resource
def load_dataset():
    return pd.read_csv("cleaned_flights.csv")

df_train = load_dataset()


# ============================
# Chargement modèles
# ============================

@st.cache_resource
def load_all():
    clf = joblib.load("best_rf_model.pkl")
    reg = joblib.load("best_reg_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return clf, reg, scaler

clf, reg, scaler = load_all()


# ============================
# Helper features
# ============================

def prepare_features(user_dict):
    template = {
        'depdelay': 0,
        'origin': 'ATL',
        'dest': 'DFW',
        'uniquecarrier': 'AA',
        'scheduledhour': 15,
        'tailnum': 'N12345',
        'numflights': 10,
        'origincityname': 'Atlanta',
        'windspeed': 10,
        'windgustdummy': 0,
        'raindummy': 0,
        'snowdummy': 0,
        'is_holiday_season': 0
    }

    map_cols = {
        "Retard départ": "depdelay",
        "Aéroport origine": "origin",
        "Aéroport destination": "dest",
        "Compagnie": "uniquecarrier",
        "Heure départ": "scheduledhour",
        "Numéro avion": "tailnum",
        "Ville origine": "origincityname",
        "Nombre vols": "numflights",
        "Vitesse vent": "windspeed",
        "Rafales vent": "windgustdummy",
        "Pluie": "raindummy",
        "Neige": "snowdummy",
        "Saison vacances": "is_holiday_season"
    }

    for ui_name, real_name in map_cols.items():
        template[real_name] = user_dict[ui_name]

    return pd.DataFrame([template])


# ============================
# Interface utilisateur
# ============================

st.header("Entrez les informations du vol")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Informations Vol")

    comp_choice = st.selectbox(
        "Compagnie Aérienne",
        unique_vals["uniquecarrier"] + ["Autre (manuel)"]
    )
    compagnie = st.text_input("Entrez la compagnie :", "") if comp_choice == "Autre (manuel)" else comp_choice

    org_choice = st.selectbox(
        "Aéroport de Départ",
        unique_vals["origin"] + ["Autre (manuel)"]
    )
    origine = st.text_input("Entrez l'aéroport d'origine :", "") if org_choice == "Autre (manuel)" else org_choice

    dest_choice = st.selectbox(
        "Aéroport d'Arrivée",
        unique_vals["dest"] + ["Autre (manuel)"]
    )
    destination = st.text_input("Entrez l'aéroport d'arrivée :", "") if dest_choice == "Autre (manuel)" else dest_choice

    heure_depart = st.slider("Heure de Départ", 0, 23, 15)
    retard_depart = st.number_input("Retard au Départ (min)", -60, 300, 0)


with col2:
    st.subheader("Informations Avion / Ville")

    tn_choice = st.selectbox(
        "Numéro d'avion (tailnum)",
        unique_vals["tailnum"] + ["Autre (manuel)"]
    )
    tailnum = st.text_input("Entrez tailnum :", "") if tn_choice == "Autre (manuel)" else tn_choice

    oc_choice = st.selectbox(
        "Ville d'origine",
        unique_vals["origincityname"] + ["Autre (manuel)"]
    )
    origincityname = st.text_input("Entrez la ville :", "") if oc_choice == "Autre (manuel)" else oc_choice

    numflights = st.number_input("Nombre total de vols de la compagnie", 0, 201, 16)

    st.subheader("Conditions Météo")

    vent = st.slider("Vitesse du Vent (km/h)", 0, 100, 10)
    rafales_vent = st.checkbox("Rafales de Vent")
    pluie = st.checkbox("Pluie")
    neige = st.checkbox("Neige")
    saison_vacances = st.checkbox("Saison de Vacances")


# ============================
# Prédiction
# ============================

if st.button("Prédire le Retard", type="primary"):

    user_data = {
        "Retard départ": retard_depart,
        "Aéroport origine": origine,
        "Aéroport destination": destination,
        "Compagnie": compagnie,
        "Heure départ": heure_depart,
        "Numéro avion": tailnum,
        "Ville origine": origincityname,
        "Nombre vols": numflights,
        "Vitesse vent": vent,
        "Rafales vent": 1 if rafales_vent else 0,
        "Pluie": 1 if pluie else 0,
        "Neige": 1 if neige else 0,
        "Saison vacances": 1 if saison_vacances else 0
    }

    df = prepare_features(user_data)

    # Colonnes catégoriques
    cat_cols = ['origin', 'dest', 'uniquecarrier', 'tailnum', 'origincityname']

    # TargetEncoder directement dans Streamlit
    encoder = TargetEncoder(cols=cat_cols, handle_unknown="value")
    encoder.fit(df_train[cat_cols], df_train['arrdelay'])  # IMPORTANT

    df_cat = encoder.transform(df[cat_cols])
    df_encoded = pd.concat([df_cat, df.drop(columns=cat_cols)], axis=1)

    # Réordonner les colonnes exactement comme le scaler
    df_encoded = df_encoded[scaler.feature_names_in_]

    # Classification (scalée)
    df_scaled = scaler.transform(df_encoded)
    class_pred = clf.predict(df_scaled)[0]
    prob = clf.predict_proba(df_scaled)[0][1]

    # Régression
    reg_delay = reg.predict(df_encoded)[0]

    # ============================
    # Affichage résultats
    # ============================

    st.subheader(" Résultats de la Prédiction")

    c1, c2 = st.columns(2)

    with c1:
        if class_pred == 1:
            st.error("Retard ≥ 15 min prédit")
        else:
            st.success("Vol à l'heure")

        st.metric("Probabilité de retard", f"{prob*100:.1f}%")

        # Afficher retard estimé comme intervalle ±15 min (borne inférieure clampée à 0)
        interval = 15
        lower = max(0.0, reg_delay - interval)
        cal_delay = max(0.0, reg_delay)
        upper = reg_delay + interval
        st.metric("Retard estimé", f"{cal_delay:.1f} min")
        st.write(f"Intervalle estimé : {lower:.1f} – {upper:.1f} min (±{interval} min)")

    with c2:
        fig, ax = plt.subplots(figsize=(8, 1))
        ax.barh([0], [prob], color='red' if prob > 0.5 else 'green')
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Probabilité de retard")
        ax.axvline(0.5, color='black', linestyle='--')
        st.pyplot(fig)
