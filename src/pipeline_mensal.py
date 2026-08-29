"""Pipeline de scoring mensal: baixa o CSV mais recente da ANP, aplica o
modelo ja treinado (sem retreinar) e gera o JSON consumido pelo mapa em
docs/index.html.

Uso:
    python src/pipeline_mensal.py                       # detecta o mes mais recente sozinho
    python src/pipeline_mensal.py --ano 2025 --mes 12    # forca um mes especifico

Pensado para rodar tanto localmente (sua maquina) quanto no GitHub Actions
(.github/workflows/atualizar_mapa.yml, agendado para todo dia 5 do mes),
reaproveitando os artefatos de producao salvos em models/ pelo notebook 04
(referencia_precos.joblib, preprocessador.joblib, melhor_modelo.joblib).

VISAO GERAL DO FLUXO (para quem esta vendo isso pela primeira vez):

    ANP (site)  --(1. download)-->  CSV do mes
                                        |
                                (2. limpeza, igual ao notebook 01)
                                        |
                                (3. features + modelo JA TREINADOS)
                                        |
                                (4. agregacao por estado + ranking nacional)
                                        |
                                        v
                            docs/dados/latest.json  --> consumido pelo
                                                         mapa em docs/index.html

Este arquivo e dividido em blocos grandes (BLOCO 1, 2, 3...) na ordem em
que o `main()` os usa, cada um com objetivo e racional explicados antes do
codigo — a ideia e dar para ler de cima para baixo como um roteiro de aula.
"""


########################################################################
# BLOCO 1 — CONFIGURACAO DO AMBIENTE E COMPATIBILIDADE COM O GITHUB ACTIONS
########################################################################
# Objetivo: deixar prontas as bibliotecas e os ajustes de ambiente ANTES de
# qualquer logica de negocio do pipeline.
#
# Racional: este script roda em dois lugares bem diferentes — a maquina
# local (onde tudo "sempre funciona") e o runner do GitHub Actions (uma
# maquina efemera na nuvem, recriada do zero a cada execucao). Duas
# pegadinhas reais que so aparecem no runner e que precisam ser corrigidas
# aqui, no topo do arquivo, antes de qualquer requisicao de rede:
#
#   1) o runner do GitHub Actions frequentemente NAO tem rota IPv6
#      funcional para a internet, mesmo quando o sistema operacional
#      "acha" que tem. O site da ANP (www.gov.br) tem endereco IPv6, e a
#      biblioteca `requests` tenta esse caminho primeiro por padrao — o
#      resultado e "Network is unreachable", um erro dificil de entender
#      se voce nao sabe que e um problema de infraestrutura do runner, e
#      nao do seu codigo. A correcao: forcar a biblioteca de rede
#      (urllib3, usada por baixo dos panos pelo `requests`) a so tentar
#      IPv4.
#   2) `src/pipeline_mensal.py` fica DENTRO da pasta `src/`, mas precisa
#      importar outros modulos como se `src` fosse um pacote visto da raiz
#      do projeto (`from src.data import ...`). Isso so funciona se a raiz
#      do projeto estiver no `sys.path` — por isso o `sys.path.insert`
#      logo abaixo, antes de qualquer `from src.xxx import ...`.
########################################################################
import argparse
import json
import socket
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import requests
import urllib3.util.connection as _urllib3_conn

# Correcao 1: forcar IPv4 (ver explicacao do BLOCO 1 acima).
_urllib3_conn.allowed_gai_family = lambda: socket.AF_INET

# Correcao 2: colocar a raiz do projeto no sys.path (ver explicacao acima).
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))


########################################################################
# BLOCO 2 — CONSTANTES E PARAMETROS DO PROJETO
########################################################################
# Objetivo: centralizar, em um unico lugar, tudo que pode mudar sem exigir
# reescrever a logica do pipeline — a URL de onde os dados vem, e os nomes
# "bonitos" dos estados que aparecem na pagina publica.
#
# Racional: `URL_BASE` usa um padrao fixo (ano + mes com dois digitos) que
# a propria ANP mantem estavel mes a mes — descobrimos esse padrao
# investigando o site manualmente, e ele e o unico lugar do codigo que
# "conhece" a estrutura de pastas da ANP. Se um dia a ANP mudar o
# endereco, so este trecho precisa ser ajustado.
#
# `NOME_ESTADO` existe porque o dataset da ANP so traz a sigla do estado
# (ex.: "PI"), mas a pagina do mapa (docs/index.html) precisa do nome por
# extenso para o texto ficar legivel para quem esta vendo ("Piaui" em vez
# de so "PI"). Os nomes ficam com acentuacao correta mesmo na versao em
# ingles da pagina — nomes proprios brasileiros nao se traduzem, so se
# escrevem certo.
########################################################################
URL_BASE = (
    "https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/"
    "arquivos/shpc/dsan/{ano}/precos-gasolina-etanol-{mes:02d}.csv"
)

NOME_ESTADO = {
    "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá",
    "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo",
    "GO": "Goiás", "MA": "Maranhão", "MG": "Minas Gerais", "MS": "Mato Grosso do Sul",
    "MT": "Mato Grosso", "PA": "Pará", "PB": "Paraíba", "PE": "Pernambuco",
    "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul", "SC": "Santa Catarina",
    "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins",
}


########################################################################
# BLOCO 3 — DOWNLOAD DOS DADOS BRUTOS DA ANP
########################################################################
# Objetivo: dado um ano e um mes, tentar baixar o CSV correspondente e
# dizer (True/False) se deu certo — sem lancar excecao quando o arquivo
# simplesmente ainda nao existe.
#
# Racional: essa funcao e usada de DUAS formas diferentes mais adiante —
# no BLOCO 4, para "sondar" varios meses ate achar um que exista (e um
# 404 ali e esperado, nao um erro); e no BLOCO 6, para baixar de verdade o
# mes que sera processado. Por isso ela devolve um booleano simples em vez
# de estourar erro, deixando quem chama decidir o que fazer com o
# resultado.
#
# O `len(resp.content) < 1000` e uma segunda checagem de seguranca: a ANP
# as vezes devolve HTTP 200 com uma pagina de erro em HTML minuscula em
# vez do CSV de verdade — um CSV real desse dataset tem varios megabytes,
# entao qualquer resposta muito pequena e tratada como falha.
########################################################################
def baixar_csv(ano, mes, destino):
    """Baixa o CSV da ANP para `destino`. Devolve True se conseguiu."""
    url = URL_BASE.format(ano=ano, mes=mes)
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    if resp.status_code != 200 or len(resp.content) < 1000:
        return False
    destino.write_bytes(resp.content)
    return True


########################################################################
# BLOCO 4 — DETECCAO AUTOMATICA DO MES MAIS RECENTE DISPONIVEL
########################################################################
# Objetivo: descobrir sozinho qual e o mes mais recente que a ANP ja
# publicou, sem precisar de intervencao manual todo mes.
#
# Racional (a parte mais contraintuitiva do pipeline): seria tentador
# simplesmente usar o mes atual do calendario. O problema e que a ANP
# **nao publica em tempo real** — ja observamos, em producao, um atraso
# de ATE 8 MESES entre o mes corrente e o ultimo arquivo realmente
# disponivel no site deles. Ou seja: em agosto de 2026, o mes mais recente
# publicado podia muito bem ainda ser dezembro de 2025.
#
# Por isso esta funcao nao pergunta "que mes e hoje?" e sim "qual e o mes
# mais recente que EXISTE DE VERDADE no servidor da ANP?" — ela comeca no
# mes atual e vai voltando, mes a mes, testando com `baixar_csv` (BLOCO 3)
# ate encontrar um arquivo que realmente baixa. `max_tentativas=14` da
# margem generosa para esse atraso sem virar um loop infinito.
#
# Consequencia pratica: quando a ANP publicar um mes novo, a PROXIMA
# execucao do pipeline (o cron mensal ou um disparo manual) vai encontrar
# esse mes novo sozinha — nenhuma linha de codigo precisa mudar.
########################################################################
def detectar_mes_mais_recente(max_tentativas=14):
    """Tenta o mes atual e volta ate `max_tentativas` meses se o arquivo
    ainda nao tiver sido publicado pela ANP."""
    hoje = date.today()
    ano, mes = hoje.year, hoje.month
    tmp = RAIZ / "data" / "_tmp_deteccao.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)  # data/ nao existe em um checkout limpo (esta no .gitignore)
    for _ in range(max_tentativas):
        if baixar_csv(ano, mes, tmp):
            tmp.unlink(missing_ok=True)
            return ano, mes
        mes -= 1
        if mes == 0:
            mes, ano = 12, ano - 1
    raise RuntimeError(
        f"Nenhum dos ultimos {max_tentativas} meses (a partir de {hoje.year}-{hoje.month:02d}) "
        "tem arquivo publicado pela ANP ainda."
    )


########################################################################
# BLOCO 5 — CARREGAMENTO E LIMPEZA DOS DADOS
########################################################################
# Objetivo: transformar o CSV bruto (ja baixado) num DataFrame limpo,
# usando exatamente a mesma limpeza que os notebooks usam.
#
# Racional: a limpeza (renomear colunas, converter datas, etc.) ja existe
# em `src/data.py` e e usada pelos 4 notebooks. Este pipeline de producao
# reaproveita a MESMA funcao em vez de reescrever a logica — assim, se um
# dia a limpeza mudar (ex.: a ANP alterar uma coluna), so precisa ser
# corrigida em um lugar, e notebooks e producao continuam consistentes
# entre si. O import fica dentro da funcao (em vez de no topo do arquivo)
# so para deixar explicito, aqui no BLOCO 5, exatamente qual funcao esta
# sendo reaproveitada.
########################################################################
def carregar_dados_bruto(caminho):
    """Mesma limpeza do src.data.carregar_dados, mas a partir de um
    caminho ja baixado (evita duplicar a funcao)."""
    from src.data import carregar_dados
    return carregar_dados(caminho)


########################################################################
# BLOCO 6 — SCORING: APLICAR O MODELO JA TREINADO (SEM RETREINAR)
########################################################################
# Objetivo: pegar os dados do mes (ja baixados e limpos) e classificar
# cada registro como anomalo ou normal, usando o modelo de producao.
#
# Racional (o principio mais importante deste arquivo): este pipeline
# **nunca retreina** nada. `referencia_precos.joblib`, `preprocessador.joblib`
# e `melhor_modelo.joblib` sao os MESMOS artefatos salvos pelo notebook 04
# depois de todo o processo de tuning e avaliacao com anomalias sinteticas
# — eles representam a versao do modelo que foi validada e "aprovada".
#
# Essa separacao entre TREINO (feito uma vez, nos notebooks, com cuidado e
# validacao) e SCORING/INFERENCIA (feito todo mes, aqui, de forma rapida e
# repetida) e um principio central de sistemas de ML em producao: você
# quer poder pontuar dados novos com frequencia, sem correr o risco de
# cada execucao aprender um modelo levemente diferente e sem re-validar
# nada. Se um dia quisermos retreinar com mais meses de dado, isso deveria
# ser uma decisao explicita (rodar os notebooks de novo, comparar com o
# modelo atual, e so entao substituir os arquivos em `models/`) — nao algo
# que acontece silenciosamente a cada execucao agendada.
#
# O restante da funcao (`rodar_pipeline`) tambem cobre os BLOCOS 7 e 8
# (agregacao por estado e ranking nacional) porque, na pratica, e mais
# didatico ver a sequencia completa — download -> limpeza -> scoring ->
# agregacao — dentro de uma unica funcao coesa.
########################################################################
def rodar_pipeline(ano, mes):
    from src.evaluation import calcular_metricas  # noqa: F401 (mantido para simetria com os notebooks)
    from src.models import prever_anomalias

    # --- 6.1: garantir que o CSV do mes pedido esta em disco -----------
    caminho_csv = RAIZ / "data" / f"precos_combustiveis_anp_{ano}-{mes:02d}.csv"
    caminho_csv.parent.mkdir(exist_ok=True)
    if not caminho_csv.exists():
        ok = baixar_csv(ano, mes, caminho_csv)
        if not ok:
            raise RuntimeError(f"Nao foi possivel baixar o CSV de {ano}-{mes:02d}.")

    df = carregar_dados_bruto(caminho_csv)  # BLOCO 5

    # --- 6.2: carregar os artefatos de producao JA TREINADOS ------------
    # Estes 3 arquivos sao o unico "estado" que este pipeline preserva de
    # execucao para execucao — sao commitados no repositorio (excecao
    # deliberada no .gitignore) exatamente para que o GitHub Actions tenha
    # acesso a eles sem precisar re-treinar nada.
    referencia = joblib.load(RAIZ / "models" / "referencia_precos.joblib")
    preprocessador = joblib.load(RAIZ / "models" / "preprocessador.joblib")
    modelo = joblib.load(RAIZ / "models" / "melhor_modelo.joblib")

    # --- 6.3: mesma engenharia de features do notebook 02 ---------------
    # `referencia.transform` calcula o desvio percentual de cada preco em
    # relacao a mediana do MESMO estado + produto (a feature mais
    # importante do projeto, explicada em detalhe na EDA); o
    # `preprocessador` aplica a mesma padronizacao/one-hot ajustada no
    # treino. Nenhum dos dois e reajustado aqui — so `.transform`.
    df_feat = referencia.transform(df)
    X = preprocessador.transform(df_feat)

    # --- 6.4: a previsao propriamente dita -------------------------------
    rotulo, score = prever_anomalias(modelo, X)

    df_feat = df_feat.copy()
    df_feat["anomalia"] = rotulo
    df_feat["score_anomalia"] = score

    ####################################################################
    # BLOCO 7 — AGREGACAO POR ESTADO (O QUE ALIMENTA O MAPA)
    ####################################################################
    # Objetivo: resumir milhares de linhas individuais em UM registro por
    # estado — exatamente o formato que o mapa (docs/index.html) precisa
    # para colorir cada estado e mostrar o "pior caso" ao clicar nele.
    #
    # Racional: a pagina do mapa nao le o CSV inteiro (seria pesado e
    # desnecessario no navegador) — ela le um JSON pequeno e ja resumido.
    # Esse resumo e calculado aqui, no servidor/CI, uma vez por mes, nao a
    # cada visita a pagina.
    ####################################################################
    por_estado = {}
    for uf, grupo in df_feat.groupby("estado"):
        piores = grupo.sort_values("score_anomalia", ascending=False).head(1)
        pior = piores.iloc[0]
        por_estado[uf] = {
            "nome": NOME_ESTADO.get(uf, uf),
            "registros": int(len(grupo)),
            "anomalias": int(grupo["anomalia"].sum()),
            "pct_anomalias": round(float(grupo["anomalia"].mean()) * 100, 2),
            "pior_caso": {
                "municipio": pior["municipio"],
                "revenda": pior["revenda"],
                "produto": pior["produto"],
                "valor_venda": float(pior["valor_venda"]),
                "score": round(float(pior["score_anomalia"]), 4),
            },
        }

    ####################################################################
    # BLOCO 8 — RANKING NACIONAL DAS MAIORES ANOMALIAS (TOP 15)
    ####################################################################
    # Objetivo: alem do resumo por estado, montar uma lista unica com as
    # 15 anomalias de maior score em todo o Brasil, para a tabela que
    # aparece embaixo do mapa.
    #
    # Racional do `.drop_duplicates(...)`: o mesmo posto pode ser
    # pesquisado mais de uma vez no mesmo mes pela ANP. Sem remover essas
    # repeticoes, o "top 15" corria o risco de mostrar o MESMO posto 4 ou
    # 5 vezes em vez de 15 casos realmente diferentes — o que e menos
    # informativo (e menos interessante) para quem esta olhando o mapa.
    ####################################################################
    top_nacional = (
        df_feat.sort_values("score_anomalia", ascending=False)
        .drop_duplicates(subset=["estado", "municipio", "revenda", "produto"])
        .head(15)[["estado", "municipio", "revenda", "produto", "valor_venda", "score_anomalia"]]
        .rename(columns={"score_anomalia": "score"})
        .to_dict(orient="records")
    )

    resumo = {
        "mes_referencia": f"{ano}-{mes:02d}",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_registros": int(len(df_feat)),
        "total_anomalias": int(df_feat["anomalia"].sum()),
        "pct_anomalias": round(float(df_feat["anomalia"].mean()) * 100, 2),
        "por_estado": por_estado,
        "top_nacional": top_nacional,
    }
    return resumo


########################################################################
# BLOCO 9 — HISTORICO MENSAL (SERIE TEMPORAL SEM BANCO DE DADOS)
########################################################################
# Objetivo: alem do mes atual, manter tambem um pequeno historico
# (mes -> totais) que alimenta o grafico de tendencia da pagina.
#
# Racional: este projeto nao usa banco de dados — o "estado" inteiro do
# pipeline mora num unico arquivo JSON versionado no git
# (`docs/dados/latest.json`). Para nao perder o historico a cada execucao
# (o `resumo` do BLOCO 6-8 so tem dados do mes atual), esta funcao le o
# JSON anterior antes de sobrescreve-lo, junta a entrada nova, e devolve
# tudo junto. Usar o `mes_referencia` como chave do dicionario garante que
# rodar o pipeline duas vezes para o mesmo mes apenas ATUALIZA aquele mes
# no historico, em vez de duplicar a entrada.
########################################################################
def atualizar_historico(resumo, caminho_saida):
    """Le o JSON existente (se houver) e atualiza o mes atual, preservando
    o historico de meses anteriores para o grafico de tendencia."""
    if caminho_saida.exists():
        anterior = json.loads(caminho_saida.read_text(encoding="utf-8"))
        historico = {h["mes_referencia"]: h for h in anterior.get("historico", [])}
    else:
        historico = {}

    entrada_historico = {
        "mes_referencia": resumo["mes_referencia"],
        "total_registros": resumo["total_registros"],
        "total_anomalias": resumo["total_anomalias"],
        "pct_anomalias": resumo["pct_anomalias"],
    }
    historico[resumo["mes_referencia"]] = entrada_historico
    resumo["historico"] = sorted(historico.values(), key=lambda h: h["mes_referencia"])
    return resumo


########################################################################
# BLOCO 10 — PONTO DE ENTRADA (CLI) E ORQUESTRACAO GERAL
########################################################################
# Objetivo: amarrar todos os blocos anteriores na ordem certa, e permitir
# rodar o script tanto "no automatico" (sem argumentos, detecta o mes
# sozinho — BLOCO 4) quanto "forcado" (com --ano/--mes, util para testar
# um mes especifico ou reprocessar um mes antigo).
#
# Racional do argparse: ter as duas formas de uso no mesmo script evita
# duplicar codigo entre "modo producao" (GitHub Actions, sem argumentos)
# e "modo debug local" (voce testando um mes especifico na sua maquina) —
# e exatamente assim que este pipeline foi validado localmente antes de
# ser publicado no workflow do GitHub Actions.
########################################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ano", type=int, default=None)
    parser.add_argument("--mes", type=int, default=None)
    args = parser.parse_args()

    if args.ano and args.mes:
        ano, mes = args.ano, args.mes
    else:
        ano, mes = detectar_mes_mais_recente()  # BLOCO 4

    print(f"Processando referencia {ano}-{mes:02d}...")
    resumo = rodar_pipeline(ano, mes)  # BLOCOS 5, 6, 7 e 8

    caminho_saida = RAIZ / "docs" / "dados" / "latest.json"
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    resumo = atualizar_historico(resumo, caminho_saida)  # BLOCO 9

    caminho_saida.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvo em {caminho_saida}")
    print(f"Total de anomalias: {resumo['total_anomalias']} / {resumo['total_registros']} ({resumo['pct_anomalias']}%)")


if __name__ == "__main__":
    sys.exit(main())
