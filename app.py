# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import requests
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

# ---------- Configuração ----------
st.set_page_config(page_title="MapaTurismo - GUI", layout="wide")
st.title("🗺️ MapaTurismo — Previsão de Potencial Turístico")

st.markdown(
    """
    **Funcionalidades:**  
    - Carregar um CSV com dados de entrada (mesmo formato do `model_input.csv`)  
    - Fazer predições usando **modelo local** (`tourism_model.pkl`) ou **API externa**  
    - Visualizar predições em tabela e mapa interativo  
    - Prever um único ponto via formulário
    """
)

# ---------- fonte do modelo / API ----------
st.sidebar.header("Configurações")
model_source = st.sidebar.radio("Fonte do Modelo", ("Local (arquivo .pkl)", "API (endpoint HTTP)"))

# Caminho padrão do modelo (ajusta se necessário)
default_model_path = os.path.join("data", "model_inputs", "tourism_model.pkl")
uploaded_model = None
pipeline = None

if model_source == "Local (arquivo .pkl)":
    st.sidebar.write("Carregar modelo local")
    use_default = st.sidebar.checkbox(f"Usar {default_model_path} (se existir)", value=True)
    if use_default and os.path.exists(default_model_path):
        try:
            pipeline = joblib.load(default_model_path)
            st.sidebar.success("Modelo carregado do disco ✅")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar modelo padrão: {e}")
            pipeline = None
    # Permitir upload do ficheiro do modelo caso queiras
    uploaded_model = st.sidebar.file_uploader("Ou carrega um arquivo .pkl do modelo", type=["pkl", "joblib"])
    if uploaded_model is not None:
        try:
            pipeline = joblib.load(uploaded_model)
            st.sidebar.success("Modelo carregado do upload ✅")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar modelo enviado: {e}")
            pipeline = None

else:
    st.sidebar.write("Configurações da API")
    api_url = st.sidebar.text_input("Endpoint da API (ex.: http://127.0.0.1:8000/predict)", value="")
    api_timeout = st.sidebar.number_input("Timeout (s)", min_value=1, max_value=60, value=10)

# ---------- prever via pipeline ou API ----------
def predict_with_pipeline(df_in):
    """Recebe DataFrame e retorna array de previsões usando pipeline local."""
    try:
        preds = pipeline.predict(df_in)
        return np.array(preds)
    except Exception as e:
        st.error(f"Erro ao predizer com o modelo local: {e}")
        return None

def predict_with_api(df_in):
    """Envia dados para API em JSON e interpreta resposta."""
    if not api_url:
        st.error("URL da API não foi fornecida.")
        return None
    payload = df_in.to_dict(orient="records")
    try:
        resp = requests.post(api_url, json=payload, timeout=api_timeout)
        if resp.status_code != 200:
            st.error(f"Resposta não OK da API: {resp.status_code} - {resp.text}")
            return None
        data = resp.json()
        # Tentar vários formatos de retorno
        if isinstance(data, dict):
            # Formato: {'predictions': [...]} ou {'potencial': ...} ou {'idh': ...}
            if "predictions" in data:
                return np.array(data["predictions"])
            # suporte a API que retorne a predição de cada registro
            if all(k in data for k in ("predictions",)) :
                return np.array(data["predictions"])
            # talvez seja lista simples
            if "result" in data:
                return np.array(data["result"])
            # caso retorne um único valor
            if "potencial" in data:
                # replicar para número de linhas
                return np.array([data["potencial"]] * len(df_in))
            # procurar chave que contenha 'idh' ou 'prediction'
            for k in data:
                if "idh" in k.lower() or "pred" in k.lower():
                    vals = data[k]
                    return np.array(vals) if isinstance(vals, (list, tuple)) else np.array([vals]*len(df_in))
        # se API retorna lista de valores diretamente
        if isinstance(data, list):
            return np.array(data)
        st.error("Formato de resposta da API não reconhecido.")
        return None
    except Exception as e:
        st.error(f"Erro na chamada à API: {e}")
        return None

# ---------- Main UI: Upload CSV / Input ----------
st.header("1. Carregar dados")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("Carrega um ficheiro CSV com as features (usar o mesmo schema de model_input.csv)", type=["csv"])
    example_csv = st.checkbox("Mostrar exemplo de colunas esperadas")
    if example_csv:
        st.caption("Exemplo de colunas (ajusta conforme o teu model_input.csv):")
        st.code(
            "nome_ponto,provincia,latitude,longitude,altitude,temperatura_media,precipitacao_anual,NDVI,EVI,NDWI,populacao,densidade,pib_per_capita,taxa_urbanizacao,distancia_cidade_km,lat_clima,outra_feature,idh"
        )

with col2:
    st.markdown("**Opções**")
    run_predict_button = st.button("Executar predição (CSV ou formulário abaixo)")

# Ler CSV se enviado
df_input = None
if uploaded is not None:
    try:
        df_input = pd.read_csv(uploaded)
        st.success(f"CSV carregado: {df_input.shape[0]} linhas, {df_input.shape[1]} colunas")
        st.dataframe(df_input.head())
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        df_input = None

# ----------Formulário de Predição Individual ----------
st.header("2. Predição individual (formulário)")
with st.form("single_form", clear_on_submit=False):
    st.write("Preenche manualmente um exemplo para prever (os nomes devem corresponder às features do modelo).")
    # Campos recomendados — ajusta conforme as tuas features
    nome_ponto = st.text_input("Nome do Ponto Turístico", value="Local Exemplo")
    provincia = st.text_input("Província", value="Luanda")
    latitude = st.number_input("Latitude", value=-11.2, format="%.6f")
    longitude = st.number_input("Longitude", value=17.8, format="%.6f")
    altitude = st.number_input("Altitude (m)", value=100.0)
    temperatura = st.number_input("Temperatura média (°C)", value=25.0)
    precipitacao = st.number_input("Precipitação anual (mm)", value=800.0)
    ndvi = st.number_input("NDVI", value=0.2)
    evi = st.number_input("EVI", value=0.1)
    ndwi = st.number_input("NDWI", value=0.05)
    populacao = st.number_input("População", value=50000)
    densidade = st.number_input("Densidade", value=300)
    pib_per_capita = st.number_input("PIB per capita", value=2000.0)
    taxa_urbanizacao = st.number_input("Taxa de urbanização (%)", value=45.0)
    distancia_cidade = st.number_input("Distância até cidade (km)", value=50.0)
    submitted = st.form_submit_button("Adicionar à tabela de predição")

    if submitted:
        single = {
            "nome_ponto": nome_ponto,
            "provincia": provincia,
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "temperatura_media": temperatura,
            "precipitacao_anual": precipitacao,
            "NDVI": ndvi,
            "EVI": evi,
            "NDWI": ndwi,
            "populacao": populacao,
            "densidade": densidade,
            "pib_per_capita": pib_per_capita,
            "taxa_urbanizacao": taxa_urbanizacao,
            "distancia_cidade_km": distancia_cidade,
        }
        # transformar para DataFrame de uma linha
        df_single = pd.DataFrame([single])
        st.success("Exemplo adicionado. Vais poder predizer quando clicares em 'Executar predição'.")
        # concatenar com df_input se existir
        if df_input is None:
            df_input = df_single
        else:
            # manter ordem de colunas do df_input se possível, caso contrário juntar
            df_input = pd.concat([df_input, df_single], ignore_index=True, sort=False)

# ---------- Executar predição ----------
if run_predict_button:
    if df_input is None or df_input.shape[0] == 0:
        st.warning("Nenhum dado para predizer. Carrega um CSV ou preenche o formulário.")
    else:
        st.info("A processar predições...")
        # Preparar dados — remover coluna target se presente
        df_for_pred = df_input.copy()
        if "idh" in df_for_pred.columns:
            df_for_pred = df_for_pred.drop(columns=["idh"])
        # Escolher fonte
        preds = None
        if model_source == "Local (arquivo .pkl)":
            if pipeline is None:
                st.error("Modelo local não carregado. Carrega o .pkl no sidebar ou descecarrega o caminho correto.")
            else:
                try:
                    preds = predict_with_pipeline(df_for_pred)
                except Exception as e:
                    st.error(f"Erro durante predição local: {e}")
        else:
            # API
            preds = predict_with_api(df_for_pred)

        if preds is not None:
            df_results = df_input.copy()
            df_results["pred_idh"] = preds
            st.success("Predições concluídas ✅")
            st.subheader("Resultados")
            st.dataframe(df_results)
            # permitir download do CSV com predições
            to_download = df_results.to_csv(index=False).encode("utf-8")
            st.download_button("Descarregar resultados (CSV)", data=to_download, file_name="predicoes_mapa_turismo.csv", mime="text/csv")
            
            # Mostrar mapa se latitude/longitude existirem
            if {'latitude', 'longitude'}.issubset(df_results.columns):
                st.subheader("Mapa das predições")
                # Centro inicial do mapa: média das coordenadas
                try:
                    lat_mean = df_results['latitude'].astype(float).mean()
                    lon_mean = df_results['longitude'].astype(float).mean()
                    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=6)
                    marker_cluster = MarkerCluster().add_to(m)
                    for _, row in df_results.iterrows():
                        try:
                            popup = f"{row.get('nome_ponto', '')}<br>Pred IDH: {row.get('pred_idh', np.nan):.3f}"
                            folium.Marker(location=[float(row['latitude']), float(row['longitude'])], popup=popup).add_to(marker_cluster)
                        except Exception:
                            continue
                    st_folium(m, width=900, height=600)
                except Exception as e:
                    st.error(f"Erro ao criar mapa: {e}")
            else:
                st.info("Colunas 'latitude' e 'longitude' não foram encontradas — o mapa não será exibido.")
        else:
            st.error("Não foi possível obter predições.")

# ---------- Mostrar importância das variáveis (se houver modelo local) ----------
st.header("3. Análise do Modelo")
if pipeline is not None:
    try:
        # tentar extrair importances e feature names
        preproc = pipeline.named_steps.get("preprocessor", None)
        model_step = pipeline.named_steps.get("model", None)
        if preproc is not None and model_step is not None:
            # get_feature_names_out pode falhar em scikit-learn antigo; usar try
            try:
                feature_names = preproc.get_feature_names_out()
            except Exception:
                # tentar construir nomes manualmente
                feature_names = []
                for name, trans, cols in preproc.transformers_:
                    if name == "num":
                        feature_names.extend(cols)
                    elif name == "cat":
                        # can't get onehot names easily — approximate
                        feature_names.extend(cols)
            importances = model_step.feature_importances_
            df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
            df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
            st.subheader("Importância das features (modelo local)")
            st.dataframe(df_imp.head(30))
        else:
            st.info("O pipeline não contém pré-processador ou modelo acessíveis.")
    except Exception as e:
        st.error(f"Erro ao mostrar importâncias: {e}")
else:
    st.info("Modelo local não carregado — não é possível mostrar importâncias.")

# ---------- Footer / dicas ----------
st.markdown("---")
st.markdown(
    """
    **Dicas:**  
    - Se a tua API retornar o output com outra chave (por exemplo `predictions`, `idh`, `potencial`), o app tenta detectar automaticamente.  
    - Para melhores previsões, garante que o CSV enviado tem as mesmas colunas usadas no treino (nomes e tipos).  
    - Se precisares, eu posso adaptar o formulário para as tuas colunas exatas (diz quais são).
    """
)