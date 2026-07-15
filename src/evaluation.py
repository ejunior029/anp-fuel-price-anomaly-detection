"""Avaliacao de modelos de anomalia sem rotulo real.

O dataset da ANP nao tem rotulo de "isto e uma anomalia". Para poder medir
precision/recall (metrica principal do projeto) injetamos anomalias
sinteticas no conjunto de teste: alteramos drasticamente o preco de uma
fracao pequena e aleatoria de postos (simulando erro de digitacao ou preco
fora da realidade de mercado) e guardamos quais linhas foram alteradas como
rotulo verdadeiro (y_true). Isso e feito SOMENTE no teste, nunca no treino.
"""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def injetar_anomalias_sinteticas(df, frac=0.02, fator_min=2.0, fator_max=4.0, seed=42):
    """Multiplica (ou divide) o valor_venda de uma fracao aleatoria de linhas.

    Devolve (df_alterado, y_true), onde y_true=1 marca as linhas alteradas.
    """
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True).copy()
    n = len(df)
    n_anomalias = max(1, int(n * frac))
    idx_anomalias = rng.choice(n, size=n_anomalias, replace=False)

    fatores = rng.uniform(fator_min, fator_max, size=n_anomalias)
    sinais = rng.choice([-1, 1], size=n_anomalias)
    # sinal +1 -> preco anormalmente alto (multiplica); sinal -1 -> preco anormalmente baixo (divide)
    ajuste = np.where(sinais == 1, fatores, 1 / fatores)

    df.loc[idx_anomalias, 'valor_venda'] = df.loc[idx_anomalias, 'valor_venda'] * ajuste

    y_true = np.zeros(n, dtype=int)
    y_true[idx_anomalias] = 1
    return df, y_true


def calcular_metricas(y_true, y_pred_anomalia, score_anomalia=None):
    """y_pred_anomalia: 1 = anomalia previsto. score_anomalia: quanto maior, mais anomalo."""
    metricas = {
        'precision': precision_score(y_true, y_pred_anomalia, zero_division=0),
        'recall': recall_score(y_true, y_pred_anomalia, zero_division=0),
        'f1': f1_score(y_true, y_pred_anomalia, zero_division=0),
    }
    if score_anomalia is not None:
        metricas['roc_auc'] = roc_auc_score(y_true, score_anomalia)
        metricas['pr_auc'] = average_precision_score(y_true, score_anomalia)
    return metricas
