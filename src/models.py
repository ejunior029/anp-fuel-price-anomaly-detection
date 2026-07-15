"""Modelos de deteccao de anomalias (nao supervisionados) a serem comparados."""
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def construir_modelos(contamination=0.02, random_state=42):
    """Devolve {nome: modelo} com hiperparametros comparaveis entre si.

    `contamination`/`nu` e a fracao esperada de anomalias nos dados; e o
    principal hiperparametro comum aos quatro modelos.

    LocalOutlierFactor usa novelty=True para permitir fit no treino e
    predict/decision_function em dados novos (por padrao o LOF so suporta
    fit_predict no mesmo conjunto usado no ajuste).
    """
    return {
        'IsolationForest': IsolationForest(
            contamination=contamination, n_estimators=200, random_state=random_state,
        ),
        'LocalOutlierFactor': LocalOutlierFactor(
            contamination=contamination, novelty=True, n_neighbors=35,
        ),
        'OneClassSVM': OneClassSVM(nu=contamination, kernel='rbf', gamma='scale'),
        'EllipticEnvelope': EllipticEnvelope(contamination=contamination, random_state=random_state),
    }


def prever_anomalias(modelo, X):
    """Roda o modelo em X e devolve (rotulo_anomalia, score_anomalia).

    rotulo_anomalia: 1 = anomalia, 0 = normal (convertido do -1/1 do sklearn).
    score_anomalia: quanto MAIOR, mais anomalo (convertido do decision_function,
    que no sklearn e positivo para pontos normais e negativo para anomalias).
    """
    rotulo_bruto = modelo.predict(X)
    rotulo_anomalia = (rotulo_bruto == -1).astype(int)
    score_anomalia = -modelo.decision_function(X)
    return rotulo_anomalia, score_anomalia
