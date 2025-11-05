# Plano Experimental do Modelo 
Caso de Uso,    Modelo,         Input             Output          Mética       Validação
-------------|---------|--------------------|------------------|----------|------------------|
Potencial    |Random   |Acessibilidade,     |Classe            |Accuracy, |Cross-Validatinon |
Turístico    |Forest   |Infraestrutura,     |(Baixo/Médio/Alto)|AUC       |Espacial          |
             |         |Biodiversidade      |                  |          |                  |
-------------|---------|--------------------|------------------|----------|------------------|
Crescimento  |LSTM     |Série temporal de   |Nº Visitantes     |RMSE, MAE |Train/Test Split  |
de Visitantes|         |visitas             |Futuros           |          |Temporal          |
-------------|---------|--------------------|------------------|----------|------------------|
Risco        |Regreção |LULC, Densidade     |Índice de Risco   |R²        |Moran’s I /       |
Ambiental    |Espacial |Populacional        |                  |          |Spatial CV        |

# Nota:“Serão usados dados reais quando disponíveis e dados sintéticos simulados para calibração inicial.”