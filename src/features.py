"""Engenharia de features para deteccao de anomalias nos precos de combustiveis.

Importante (data leakage): as medianas de referencia usadas como feature
("o quanto este preco desvia da mediana da regiao/produto") sao calculadas
SOMENTE com dados de treino (`ReferenciaPrecos.fit`) e depois aplicadas ao
teste (`.transform`). Da mesma forma, o StandardScaler/OneHotEncoder do
preprocessador so podem ser ajustados (`fit`) no treino.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUNAS_NUMERICAS = ['valor_venda', 'desvio_pct_estado_produto', 'desvio_pct_brasil_produto']
COLUNAS_CATEGORICAS = ['produto', 'regiao']


class ReferenciaPrecos:
    """Guarda medianas de preco por (estado, produto) e por produto, calculadas no treino."""

    def __init__(self):
        self.mediana_estado_produto_ = None
        self.mediana_brasil_produto_ = None

    def fit(self, df_treino):
        """Calcula e guarda as medianas de preco a partir do DataFrame de treino."""
        self.mediana_estado_produto_ = (
            df_treino.groupby(['estado', 'produto'])['valor_venda']
            .median()
            .rename('mediana_estado_produto')
            .reset_index()
        )
        self.mediana_brasil_produto_ = (
            df_treino.groupby('produto')['valor_venda']
            .median()
            .rename('mediana_brasil_produto')
            .reset_index()
        )
        return self

    def transform(self, df):
        """Adiciona as colunas de desvio percentual usando as medianas guardadas no fit."""
        df = df.merge(self.mediana_estado_produto_, on=['estado', 'produto'], how='left')
        df = df.merge(self.mediana_brasil_produto_, on='produto', how='left')
        # combinacao estado+produto nao vista no treino -> usa a mediana nacional do produto
        df['mediana_estado_produto'] = df['mediana_estado_produto'].fillna(df['mediana_brasil_produto'])

        df['desvio_pct_estado_produto'] = (
            (df['valor_venda'] - df['mediana_estado_produto']) / df['mediana_estado_produto']
        )
        df['desvio_pct_brasil_produto'] = (
            (df['valor_venda'] - df['mediana_brasil_produto']) / df['mediana_brasil_produto']
        )
        return df.drop(columns=['mediana_estado_produto', 'mediana_brasil_produto'])


def construir_preprocessador():
    """Cria o ColumnTransformer (padronizacao + one-hot) a ser ajustado apenas no treino."""
    return ColumnTransformer([
        ('num', StandardScaler(), COLUNAS_NUMERICAS),
        # drop='first' evita colunas dummy colineares (soma das colunas de uma
        # categoria = 1), o que deixava a matriz de covariancia do
        # EllipticEnvelope com posto deficiente.
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), COLUNAS_CATEGORICAS),
    ])
