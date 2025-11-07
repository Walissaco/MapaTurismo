# 📘 README — model_input.csv

##  Descrição Geral

O arquivo **`model_input.csv`** representa a base de dados consolidada e tratada do projeto **“Mapeamento de Locais Turísticos e Análise Preditiva de Desenvolvimento Turístico em Angola com IA”**.

Este conjunto de dados é o **resultado final da etapa de integração**, onde foram cruzadas informações **geoespaciais, climáticas, ambientais, socioeconômicas e de mobilidade**.  
Ele serve como **entrada direta para os modelos de Machine Learning**, utilizados para **prever o potencial turístico sustentável** de cada ponto turístico em Angola.

---

##  Estrutura do Dataset

Cada linha representa **um ponto turístico** de referência em Angola, com variáveis numéricas e geográficas que caracterizam o local.

| Coluna | Tipo | Descrição |
|:--------|:------|:-----------|
| **latitude** | `float` | Latitude geográfica do ponto turístico |
| **longitude** | `float` | Longitude geográfica do ponto turístico |
| **ndvi** | `float` | Índice de Vegetação por Diferença Normalizada (proxy da cobertura vegetal) |
| **evi** | `float` | Índice de Vegetação Melhorado (proxy de saúde da vegetação) |
| **ndwi** | `float` | Índice de Umidade da Superfície (proxy de presença de água) |
| **altitude** | `float` | Elevação média do terreno em metros |
| **populacao** | `int` | População estimada da região próxima ao ponto turístico |
| **idh** | `float` | Índice de Desenvolvimento Humano da região (0 a 1) |

>  *Em versões futuras, poderão ser adicionadas outras variáveis relevantes, como infraestrutura, acessibilidade, densidade populacional ou taxa de urbanização.*

---

##  Fonte e Integração dos Dados

Os dados foram consolidados a partir de múltiplas fontes reais e abertas:

| Categoria | Origem dos Dados |
|------------|------------------|
| **Clima e Uso do Solo** | Google Earth Engine (GEE) |
| **Mobilidade e Infraestrutura** | OpenStreetMap (OSM) |
| **Socioeconômico** | Instituto Nacional de Estatística (INE) e relatórios públicos |
| **Geoespacial** | Camadas de pontos turísticos em formato GeoJSON |

###  Referências utilizadas:
- [Instituto Nacional de Estatística de Angola (INE)](https://www.ine.gov.ao/)
- [Angola 2014 Census (UN Stats)](https://unstats.un.org/unsd/demographic-social/census/documents/Angola/Angola%202014%20Census.pdf)
- [Google Earth Engine](https://earthengine.google.com/)
- [OpenStreetMap](https://www.openstreetmap.org/)

---

##  Propósito Analítico

Este dataset foi projetado para:
- **Treinar modelos preditivos** de *potencial turístico*;
- **Analisar correlações** entre fatores climáticos, ambientais e sociais;
- **Gerar mapas interativos** e *insights* de desenvolvimento sustentável.

---

##  Exemplo de Visualização (trecho)

| latitude | longitude | ndvi | evi | ndwi | altitude | populacao | idh |
|:----------|:-----------|:-----|:----|:-----|:----------|:-----------|:----|
| -9.0740668 | 16.000316 | 0.8 | 0.16 | 0.02 | 287.0 | 95533 | 0.543 |
| -9.221147  | 13.090001 | 0.77 | 0.68 | 0.08 | 1148.0 | 52641 | 0.590 |
| ... | ... | ... | ... | ... | ... | ... | ... |

---

##  Uso no Pipeline

O arquivo `model_input.csv` é consumido pelo notebook:

