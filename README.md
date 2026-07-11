# Projeto: Anomalias

Detectar registros anomalos / outliers nos dados.

## Estrutura
- data/       dados brutos e processados
- notebooks/  exploracao (EDA) e experimentos
- src/        codigo reutilizavel (pipelines, funcoes)
- models/     modelos treinados salvos
- reports/    graficos e resultados

## Como comecar
1. Ativar o ambiente:  .\venv\Scripts\Activate.ps1
2. Instalar libs:      pip install -r requirements.txt
3. Abrir os notebooks em notebooks/

## Algoritmos
IsolationForest, LocalOutlierFactor, OneClassSVM

## Metricas de avaliacao
Precision/Recall nos outliers, score de anomalia
