"""Carregamento e limpeza dos dados de precos de combustiveis (ANP).

Fonte: Agencia Nacional do Petroleo, Gas Natural e Biocombustiveis (ANP)
Serie historica de levantamento de precos de revenda - gasolina e etanol.
https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis

IMPORTANTE — este modulo NAO baixa nada da internet. `carregar_dados` so
le e limpa um CSV que ja esta no seu disco; ele nao sabe (nem precisa
saber) de onde esse arquivo veio.

Para os notebooks (01 a 04), o download e MANUAL: baixe o CSV do mes que
quiser direto da ANP e salve em `data/`. CUIDADO: a ANP trocou o nome do
arquivo sem aviso a partir de 2026 — use o padrao certo pro mes desejado
(prefixo comum: .../dados-abertos/arquivos/shpc/dsan/{ano}/...):

    Ate dezembro/2025:  precos-gasolina-etanol-{mes:02d}.csv
    A partir de 2026:   {mes:02d}-dados-abertos-precos-gasolina-etanol.csv

Ex.: dezembro/2025 -> .../dsan/2025/precos-gasolina-etanol-12.csv
     julho/2026     -> .../dsan/2026/07-dados-abertos-precos-gasolina-etanol.csv

Se nenhum dos dois padroes funcionar pro mes que voce quer, a ANP
provavelmente publicou esse mes com um nome fora do padrao (ja aconteceu
em fev, abr e jun/2026) — ache o link certo direto na pagina da serie
historica e confira `EXCECOES_URL` em `src/pipeline_mensal.py`.

Só em producao (`src/pipeline_mensal.py`) esse download acontece sozinho
— e a funcao que faz isso (`baixar_csv`, com os 2 padroes + excecoes)
mora la, nao aqui.
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
    """Le o CSV bruto da ANP (ja baixado manualmente, ver topo do arquivo)
    e devolve um DataFrame limpo.

    `caminho` precisa apontar para um arquivo que ja existe em disco —
    esta funcao nao baixa nada.

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
