# Projeto: Anomalias

## Objetivo
Detectar registros anomalos / outliers nos dados.

## Stack
- Python 3, pandas, numpy, scikit-learn, matplotlib, seaborn
- Ambiente virtual em ./venv

## Estrutura de pastas
- data/       -> dados (nao versionados)
- notebooks/  -> EDA e experimentos
- src/        -> codigo reutilizavel
- models/     -> modelos salvos (.pkl / .joblib)
- reports/    -> graficos e resultados

## Algoritmos previstos
IsolationForest, LocalOutlierFactor, OneClassSVM

## Metrica principal
Precision/Recall nos outliers, score de anomalia

## Convencoes
- Comentar o codigo em portugues
- SEMPRE separar treino/teste ANTES de qualquer transformacao (evitar data leakage)
- Salvar graficos em ./reports/
- Salvar modelos treinados em ./models/
- Trabalhar em passos pequenos: EDA -> baseline -> modelos -> tuning -> avaliacao
