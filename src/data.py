"""Carregamento e limpeza dos dados de precos de combustiveis (ANP).

Fonte: Agencia Nacional do Petroleo, Gas Natural e Biocombustiveis (ANP)
Serie historica de levantamento de precos de revenda - gasolina e etanol.
https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis
"""
import pandas as pd

# Mapeia as colunas originais (em portugues, com espacos) para nomes curtos.
# As colunas de endereco (rua, numero, complemento, bairro, cep) e "Valor de
# Compra" (100% nula neste arquivo) sao descartadas por nao serem uteis para
# detectar precos anomalos.
COLUNAS = {
    'Regiao - Sigla': 'regiao',
    'Estado - Sigla': 'estado',
    'Municipio': 'municipio',
    'Revenda': 'revenda',
    'CNPJ da Revenda': 'cnpj_revenda',
    'Produto': 'produto',
    'Data da Coleta': 'data_coleta',
    'Valor de Venda': 'valor_venda',
    'Unidade de Medida': 'unidade_medida',
    'Bandeira': 'bandeira',
}


def carregar_dados(caminho):
    """Le o CSV bruto da ANP e devolve um DataFrame limpo.

    O arquivo usa ';' como separador, ',' como separador decimal e
    codificacao UTF-8 com BOM.
    """
    df = pd.read_csv(caminho, sep=';', encoding='utf-8-sig', decimal=',')
    df = df.rename(columns=COLUNAS)[list(COLUNAS.values())]
    df['data_coleta'] = pd.to_datetime(df['data_coleta'], format='%d/%m/%Y')
    df['produto'] = df['produto'].str.strip()
    df['bandeira'] = df['bandeira'].str.strip()
    df['estado'] = df['estado'].str.strip()
    return df
