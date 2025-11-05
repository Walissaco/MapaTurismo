# 1. Descrição Geral

O ficheiro `pois_socioeco_mobilidade.csv` contém dados **sintéticos e coerentes** de **mobilidade** e **indicadores socioeconómicos** relativos aos principais pontos turísticos (POIs) de Angola.  
O objetivo é apoiar estudos e **modelos de Machine Learning** para previsão do **índice de desenvolvimento turístico**, caracterização regional e planeamento territorial.

# 2. Estrutura do Dataset

| Coluna | Tipo | Unidade | Descrição |
|--------|------|----------|------------|
| poi_nome | string | — | Nome do ponto turístico (sem acentos). |
| provincia | string | — | Província onde o POI se localiza. |
| categoria | string | — | Tipo de local (praia, parque, histórico, paisagem, urbano, outro). |
| densidade_rodoviaria_km | float | km/km² | Densidade média de estradas na área do POI. |
| acesso_transporte_publico | float | 0–1 | Grau de acesso a transporte público. |
| iluminacao_noturna | float | 0–1 | Índice de luminosidade noturna (proxy de urbanização). |
| populacao_municipio | int | habitantes | População estimada do município. |
| densidade_populacional | float | hab/km² | Densidade populacional local. |
| rendimento_medio_mensal_usd | float | USD | Rendimento médio mensal. |
| taxa_emprego | float | 0–1 | Percentagem da população empregada. |
| nivel_educacao | float | 0–1 | Índice médio de educação. |
| pib_per_capita_usd | float | USD | PIB per capita estimado. |
| taxa_pobreza | float | 0–1 | Proporção da população em situação de pobreza. |
| acesso_servicos_basicos | float | 0–1 | Índice de acesso a água, energia e saúde. |
| desenvolvimento_humano | float | 0–1 | Índice sintético de desenvolvimento humano. |
| numero_visitantes_anual | int | visitantes | Estimativa anual de visitantes ao ponto turístico. |

---

# 3. Metodologia de Geração e Coleta

Os dados foram gerados com base em **modelagem sintética coerente**, ajustada por província, usando escalas derivadas de indicadores públicos.


# 4. Fontes de Referência e Calibração

As faixas de valores foram calibradas com base em dados públicos e fontes reconhecidas:

| Fonte | Tipo de dados utilizados |
|-------|---------------------------|
| **Instituto Nacional de Estatística (INE) – Angola** | Demografia, densidade populacional, taxa de pobreza |
| **World Bank Open Data (2023)** | PIB per capita, rendimento médio, acesso a serviços básicos |
| **WorldPop / UN Population Division** | Estimativas populacionais e de densidade |
| **OpenStreetMap (OSM)** | Infraestrutura viária, transporte e iluminação urbana |
| **NOAA VIIRS Nighttime Lights** | Índice de iluminação noturna como proxy de urbanização |
| **UNDP (PNUD Angola)** | Índice de Desenvolvimento Humano regional |
| **Google Maps / OpenTripMap (validação semântica)** | Verificação de categorias e relevância turística dos POIs |

---