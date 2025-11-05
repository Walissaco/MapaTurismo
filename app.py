# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import os

# ---------- Configuração ----------
st.set_page_config(page_title="MapaTurismo", layout="wide")
st.title("🗺️ MapaTurismo — Previsão de Potencial Turístico")

MODEL_PATH = os.path.join("models", "tourism_model.pkl")

# ---------- Carregar modelo ----------
model = None
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Modelo carregado com sucesso.")
except Exception as e:
    st.error(f"Erro ao carregar modelo: {e}")
    st.stop()

# ---------- Função de preparação ----------
def prepare_df_for_model(df_in, model_obj):
    df = df_in.copy()
    if "idh" in df.columns:
        df = df.drop(columns=["idh"])

    if hasattr(model_obj, "named_steps"):
        return df

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.Categorical(df[col]).codes

    if hasattr(model_obj, "feature_names_in_"):
        expected = list(model_obj.feature_names_in_)
        for c in expected:
            if c not in df.columns:
                df[c] = 0
        df = df.reindex(columns=expected, fill_value=0)
        return df

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

# ---------- Upload ----------
st.header("1️⃣ Carregar dados (CSV) ou adicionar ponto manualmente")
uploaded = st.file_uploader("Carrega um ficheiro CSV com as colunas usadas no treino (ex.: model_input.csv)", type=["csv"])

if "df_input" not in st.session_state:
    st.session_state.df_input = pd.DataFrame()
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "df_results" not in st.session_state:
    st.session_state.df_results = pd.DataFrame()

if uploaded:
    try:
        csv_df = pd.read_csv(uploaded)
        csv_df.columns = csv_df.columns.str.strip().str.lower()
        rename_map = {
            "lat": "latitude",
            "lon": "longitude",
            "lat_clima": "latitude",
            "lon_clima": "longitude"
        }
        csv_df.rename(columns=rename_map, inplace=True)
        st.session_state.df_input = csv_df
        st.success(f"CSV carregado: {csv_df.shape[0]} linhas.")
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")

# ---------- Formulário manual ----------
with st.form("form_predicao", clear_on_submit=False):
    st.write("Ou preenche manualmente um ponto:")
    nome_ponto = st.text_input("Nome do Ponto Turístico", "Local Exemplo")
    provincia = st.text_input("Província", "Luanda")
    latitude = st.number_input("Latitude", value=-11.2, format="%.6f")
    longitude = st.number_input("Longitude", value=17.8, format="%.6f")
    altitude = st.number_input("Altitude (m)", value=100.0)
    temperatura_media = st.number_input("Temperatura média (°C)", value=25.0)
    precipitacao_anual = st.number_input("Precipitação anual (mm)", value=800.0)
    NDVI = st.number_input("NDVI", value=0.2)
    EVI = st.number_input("EVI", value=0.1)
    NDWI = st.number_input("NDWI", value=0.05)
    populacao = st.number_input("População", value=50000)
    densidade = st.number_input("Densidade", value=300)
    pib_per_capita = st.number_input("PIB per capita", value=2000.0)
    taxa_urbanizacao = st.number_input("Taxa de urbanização (%)", value=45.0)
    distancia_cidade_km = st.number_input("Distância até cidade (km)", value=50.0)
    submitted = st.form_submit_button("Adicionar ponto")

if submitted:
    new_row = pd.DataFrame([{
        "nome_ponto": nome_ponto,
        "provincia": provincia,
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "temperatura_media": temperatura_media,
        "precipitacao_anual": precipitacao_anual,
        "NDVI": NDVI,
        "EVI": EVI,
        "NDWI": NDWI,
        "populacao": populacao,
        "densidade": densidade,
        "pib_per_capita": pib_per_capita,
        "taxa_urbanizacao": taxa_urbanizacao,
        "distancia_cidade_km": distancia_cidade_km
    }])
    st.session_state.df_input = pd.concat([st.session_state.df_input, new_row], ignore_index=True)
    st.success("✅ Ponto adicionado à tabela.")

# ---------- Mostrar tabela ----------
if st.session_state.df_input.empty:
    st.info("Tabela vazia — carrega um CSV ou adiciona pontos.")
else:
    st.subheader("📋 Dados carregados")
    st.dataframe(st.session_state.df_input)

# ---------- Botão de predição ----------
if st.button("🔎 Executar Predição e Mostrar Mapa"):
    df_to_pred = st.session_state.df_input.copy()
    if df_to_pred.empty:
        st.warning("Nenhum dado para predizer.")
    else:
        try:
            df_ready = prepare_df_for_model(df_to_pred, model)
            preds = model.predict(df_ready)
            df_results = df_to_pred.copy()
            df_results["pred_idh"] = preds

            st.session_state.df_results = df_results
            st.session_state.predicted = True
            st.success("✅ Predições realizadas com sucesso!")
        except Exception as e:
            st.error(f"Erro ao predizer com o modelo: {e}")

# ---------- Mostrar resultados persistentes ----------
if st.session_state.predicted and not st.session_state.df_results.empty:
    df_results = st.session_state.df_results
    st.subheader("📊 Resultados")
    st.dataframe(df_results)

    csv_bytes = df_results.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Descarregar CSV", data=csv_bytes, file_name="predicoes.csv", mime="text/csv")

    # ---------- Mapa ----------
    if {"latitude", "longitude"}.issubset(df_results.columns):
        st.subheader("🗺️ Mapa das Predições")
        df_results["latitude"] = pd.to_numeric(df_results["latitude"], errors="coerce")
        df_results["longitude"] = pd.to_numeric(df_results["longitude"], errors="coerce")

        df_results = df_results.dropna(subset=["latitude", "longitude"])
        if df_results.empty:
            st.warning("Nenhuma coordenada válida encontrada.")
        else:
            lat_mean = df_results["latitude"].mean()
            lon_mean = df_results["longitude"].mean()
            mapa = folium.Map(location=[lat_mean, lon_mean], zoom_start=6)
            cluster = MarkerCluster().add_to(mapa)

            for _, r in df_results.iterrows():
                popup = f"{r.get('nome_ponto', '')}<br>IDH Previsto: {r.get('pred_idh', np.nan):.3f}"
                folium.Marker(
                    location=[r["latitude"], r["longitude"]],
                    popup=popup
                ).add_to(cluster)

            st_folium(mapa, width=900, height=600)
    else:
        st.warning("As colunas 'latitude' e 'longitude' não foram encontradas na tabela.")

st.markdown("---")
st.caption("💡 Dica: o CSV deve conter colunas 'latitude' e 'longitude' (ou 'lat_clima'/'lon_clima').")
