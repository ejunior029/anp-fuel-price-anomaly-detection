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
    """Guarda medianas de preco por (estado, produto) e por produto, calculadas no treino.

    Por que uma CLASSE (e nao so uma funcao)? Porque este objeto precisa
    LEMBRAR de um resultado (as medianas) calculado uma vez no treino
    (`fit`) para poder reaplicar esse mesmo resultado depois, em dados
    diferentes (`transform`) — uma funcao comum "esquece" tudo assim que
    termina de rodar; um objeto guarda estado entre uma chamada e outra.
    Essa dupla `fit` (aprende) / `transform` (aplica o que aprendeu) e a
    mesma convencao dos transformadores do scikit-learn (StandardScaler,
    OneHotEncoder etc.) — por isso os nomes dos metodos sao esses.

    Convencao do sklearn adotada aqui: atributos que so existem DEPOIS do
    `fit` (ou seja, que dependem dos dados de treino) terminam com "_"
    (`mediana_estado_produto_`), pra deixar claro, so pelo nome, que eles
    nao existem logo depois de criar o objeto — so depois de chamar `fit`.
    """

    def __init__(self):
        # __init__ e o "construtor": roda automaticamente quando voce
        # escreve `ReferenciaPrecos()`. `self` e o proprio objeto sendo
        # criado — e atraves dele que um metodo guarda algo para outro
        # metodo usar depois (ex.: o que `fit` calcula aqui embaixo fica
        # disponivel em `self` para o `transform` usar mais tarde).
        # Aqui so deixamos os atributos reservados, ainda vazios (None),
        # porque as medianas de verdade so existem depois do fit.
        self.mediana_estado_produto_ = None
        self.mediana_brasil_produto_ = None

    def fit(self, df_treino):
        """Calcula e guarda (em `self`) as medianas de preco do treino."""
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
        # `return self` permite escrever `ReferenciaPrecos().fit(treino)`
        # numa linha so (cria o objeto, ajusta, e devolve o proprio objeto
        # ja ajustado) — o mesmo padrao usado pelos transformadores do
        # scikit-learn.
        return self

    def transform(self, df):
        """Usa as medianas guardadas em `self` (pelo fit) para adicionar
        as colunas de desvio percentual em `df`."""
        # `self.mediana_estado_produto_` e `self.mediana_brasil_produto_`
        # foram calculadas no fit(), com o DataFrame de TREINO. Aqui em
        # transform() elas so sao lidas (nunca recalculadas) — e assim que
        # se evita vazamento de dados: o teste (ou qualquer dado novo) e
        # sempre comparado contra a referencia do treino, nunca contra a
        # sua propria mediana.
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
