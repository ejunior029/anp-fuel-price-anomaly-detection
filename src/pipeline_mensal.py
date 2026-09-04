"""Pipeline de scoring mensal: baixa o CSV mais recente da ANP, aplica o
modelo ja treinado (sem retreinar) e gera o JSON consumido pelo mapa em
docs/index.html.

Uso:
    python src/pipeline_mensal.py                       # detecta o mes mais recente sozinho
    python src/pipeline_mensal.py --ano 2025 --mes 12    # forca um mes especifico

Fluxo: ANP (CSV) -> limpeza -> features/modelo ja treinados -> agregacao
por estado + ranking nacional -> docs/dados/latest.json -> mapa.

Dividido em blocos numerados na ordem em que o `main()` os usa, cada um
com objetivo e racional resumidos antes do codigo.
"""


########################################################################
# BLOCO 1 — AMBIENTE E COMPATIBILIDADE COM O GITHUB ACTIONS
########################################################################
# Objetivo: imports e 2 ajustes que so importam quando isto roda no
# runner do GitHub Actions (nao na sua maquina local).
#
# Racional:
#  1) o runner as vezes nao tem rota IPv6 funcional; o site da ANP tem
#     endereco IPv6 e `requests` tenta ele primeiro -> "Network is
#     unreachable". Forcamos IPv4.
#  2) este arquivo esta dentro de src/, mas importa `from src.xxx import`
#     como se a raiz do projeto fosse o pacote — por isso ela precisa
#     entrar no sys.path manualmente.
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

_urllib3_conn.allowed_gai_family = lambda: socket.AF_INET  # forca IPv4

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))


########################################################################
# BLOCO 2 — CONSTANTES DO PROJETO
########################################################################
# Objetivo: URL(s) da ANP e nomes por extenso dos estados, num so lugar.
#
# Racional dos DOIS padroes de URL: a ANP usava um nome de arquivo ate
# dezembro/2025 e trocou para outro a partir de 2026 — sem aviso, no meio
# da serie historica (`precos-gasolina-etanol-12.csv` virou
# `01-dados-abertos-precos-gasolina-etanol.csv`). Por isso tentamos os
# dois formatos, nessa ordem, em `baixar_csv`.
#
# Alem dos 2 padroes gerais, a ANP publicou pelo menos 3 meses com nomes
# de arquivo que nao seguem NENHUM padrao previsivel — erros pontuais do
# lado deles, verificados manualmente um a um (conferimos que o arquivo
# de verdade existe atras de cada link, so o nome que esta "errado"):
#   - fev/2026: erro de digitacao ("02-cados-abertos-preco-...", sem "s")
#   - abr/2026: falta a extensao ".csv" no nome do arquivo
#   - jun/2026: formato unico, com o ano-mes repetido no meio do nome
# Guardamos essas 3 URLs literais em `EXCECOES_URL` (chave = (ano, mes))
# como uma correcao pontual para um problema pontual — nao um 3º padrao
# geral, porque nao ha garantia de que o proximo mes quebrado vai seguir
# qualquer uma dessas variacoes. Um mes novo que nao bater nem nos 2
# padroes gerais nem em `EXCECOES_URL` e tratado como "pulado" (BLOCO 10),
# nao como erro fatal, para o backfill nao travar por causa de 1 mes.
#
# Nomes proprios dos estados nao se traduzem — so ficam com acento certo,
# mesmo na versao em ingles da pagina do mapa.
########################################################################
URL_PADROES = [
    "https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/"
    "arquivos/shpc/dsan/{ano}/precos-gasolina-etanol-{mes:02d}.csv",  # padrao ate dez/2025
    "https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/"
    "arquivos/shpc/dsan/{ano}/{mes:02d}-dados-abertos-precos-gasolina-etanol.csv",  # padrao a partir de 2026
]

EXCECOES_URL = {
    (2026, 2): "https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/"
               "arquivos/shpc/dsan/2026/02-cados-abertos-preco-gasolina-etanol.csv",
    (2026, 4): "https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/"
               "arquivos/shpc/dsan/2026/04-dados-abertos-precos-gasolina-etanol",
    (2026, 6): "https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/"
               "arquivos/shpc/dsan/2026/06-dados-abertos-precos-2026-06-gasolina-etanol.csv",
}

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
# BLOCO 3 — DOWNLOAD DO CSV DA ANP
########################################################################
# Objetivo: baixar o CSV de um mes e devolver True/False (sem estourar
# erro se o arquivo simplesmente ainda nao existir).
#
# Racional: usada tanto para "sondar" varios meses no BLOCO 4 (404 e
# esperado ali) quanto para o download de verdade no BLOCO 6. O tamanho
# minimo evita cair numa pagina de erro em HTML disfarcada de HTTP 200.
########################################################################
def baixar_csv(ano, mes, destino):
    """Tenta `EXCECOES_URL` (mes com nome de arquivo fora do padrao,
    conferido manualmente) e depois cada URL em `URL_PADROES`, nessa
    ordem, ate uma funcionar. Baixa o CSV da ANP para `destino`.
    Devolve True se conseguiu."""
    urls = []
    if (ano, mes) in EXCECOES_URL:
        urls.append(EXCECOES_URL[(ano, mes)])
    urls += [padrao.format(ano=ano, mes=mes) for padrao in URL_PADROES]

    for url in urls:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if resp.status_code == 200 and len(resp.content) >= 1000:
            destino.write_bytes(resp.content)
            return True
    return False


########################################################################
# BLOCO 4 — DETECTAR O MES MAIS RECENTE DISPONIVEL
########################################################################
# Objetivo: descobrir sozinho o mes mais recente ja publicado pela ANP.
#
# Racional: a ANP nao publica em tempo real — ja vimos ate 8 meses de
# atraso em producao. Por isso a funcao nao assume "mes atual"; ela testa
# de tras para frente, mes a mes, ate achar um arquivo que baixa de
# verdade. Assim, quando a ANP publicar um mes novo, a proxima execucao
# encontra sozinha, sem mudar codigo.
########################################################################
def detectar_mes_mais_recente(max_tentativas=14):
    """Tenta o mes atual e volta ate `max_tentativas` meses se o arquivo
    ainda nao tiver sido publicado pela ANP."""
    hoje = date.today()
    ano, mes = hoje.year, hoje.month
    tmp = RAIZ / "data" / "_tmp_deteccao.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)  # data/ nao existe em um checkout limpo
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
# BLOCO 5 — CARREGAR E LIMPAR OS DADOS
########################################################################
# Objetivo: transformar o CSV baixado num DataFrame limpo.
#
# Racional: reaproveita `src/data.py`, a mesma limpeza usada pelos 4
# notebooks — se a limpeza mudar um dia, muda num lugar so.
########################################################################
def carregar_dados_bruto(caminho):
    """Mesma limpeza do src.data.carregar_dados, mas a partir de um
    caminho ja baixado (evita duplicar a funcao)."""
    from src.data import carregar_dados
    return carregar_dados(caminho)


########################################################################
# BLOCO 6 — SCORING COM O MODELO JA TREINADO (SEM RETREINAR)
########################################################################
# Objetivo: classificar cada registro do mes como anomalo ou normal.
#
# Racional: `referencia_precos`, `preprocessador` e `melhor_modelo` sao os
# artefatos salvos pelo notebook 04 depois do tuning/avaliacao — a versao
# "aprovada". Este pipeline so aplica (`.transform`/`.predict`), nunca
# reajusta: treino (notebooks, com validacao) e inferencia (aqui, mensal
# e automatico) ficam separados de proposito, para nao introduzir drift
# silencioso a cada execucao. Retreinar deveria ser uma decisao explicita,
# nao um efeito colateral do cron.
#
# Esta funcao tambem cobre os BLOCOS 7 e 8 (agregacao e ranking), por
# ficar mais didatico ver a sequencia inteira junta.
########################################################################
def rodar_pipeline(ano, mes):
    from src.evaluation import calcular_metricas  # noqa: F401 (mantido para simetria com os notebooks)
    from src.models import prever_anomalias

    caminho_csv = RAIZ / "data" / f"precos_combustiveis_anp_{ano}-{mes:02d}.csv"
    caminho_csv.parent.mkdir(exist_ok=True)
    if not caminho_csv.exists():
        ok = baixar_csv(ano, mes, caminho_csv)
        if not ok:
            raise RuntimeError(f"Nao foi possivel baixar o CSV de {ano}-{mes:02d}.")

    df = carregar_dados_bruto(caminho_csv)  # BLOCO 5

    # Artefatos de producao — commitados de proposito (excecao no
    # .gitignore) para o GitHub Actions ter acesso sem re-treinar nada.
    referencia = joblib.load(RAIZ / "models" / "referencia_precos.joblib")
    preprocessador = joblib.load(RAIZ / "models" / "preprocessador.joblib")
    modelo = joblib.load(RAIZ / "models" / "melhor_modelo.joblib")

    df_feat = referencia.transform(df)  # mesma feature de desvio vs. mediana do notebook 02
    X = preprocessador.transform(df_feat)
    rotulo, score = prever_anomalias(modelo, X)

    df_feat = df_feat.copy()
    df_feat["anomalia"] = rotulo
    df_feat["score_anomalia"] = score

    ####################################################################
    # BLOCO 7 — AGREGACAO POR ESTADO (ALIMENTA O MAPA)
    ####################################################################
    # Objetivo: resumir as linhas num registro por estado.
    # Racional: o mapa le um JSON pequeno ja resumido, nao o CSV inteiro.
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
    # BLOCO 8 — RANKING NACIONAL (TOP 15)
    ####################################################################
    # Objetivo: as 15 maiores anomalias do Brasil, para a tabela do mapa.
    # Racional do drop_duplicates: o mesmo posto pode ser pesquisado mais
    # de uma vez no mes — sem isso, o top 15 repetiria o mesmo caso.
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
# Objetivo: manter um historico mes -> totais para o grafico de tendencia.
#
# Racional: nao ha banco de dados — o "estado" mora no proprio JSON
# versionado no git. Por isso le o JSON anterior antes de sobrescrever, e
# usa `mes_referencia` como chave para atualizar (nao duplicar) o mes.
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


def ultimo_mes_no_historico(caminho_saida):
    """Devolve (ano, mes) do mes mais recente ja salvo no historico, ou
    None se o arquivo ainda nao existe (primeira execucao)."""
    if not caminho_saida.exists():
        return None
    dados = json.loads(caminho_saida.read_text(encoding="utf-8"))
    historico = dados.get("historico", [])
    if not historico:
        return None
    mais_recente = max(h["mes_referencia"] for h in historico)
    ano, mes = mais_recente.split("-")
    return int(ano), int(mes)


def proximo_mes(ano, mes):
    """(ano, mes) seguinte, com virada de dezembro para janeiro do ano seguinte."""
    mes += 1
    if mes == 13:
        mes, ano = 1, ano + 1
    return ano, mes


def meses_faltantes(ultimo_salvo, mais_recente_anp):
    """Lista, em ordem cronologica, os meses entre `ultimo_salvo` (exclusive)
    e `mais_recente_anp` (inclusive) — o "buraco" a preencher no backfill.

    Se `ultimo_salvo` for None (primeira execucao) ou ja estiver em dia,
    devolve so o `mais_recente_anp`, preservando o comportamento normal de
    "um mes por execucao" no caso comum (sem atraso acumulado)."""
    if ultimo_salvo is None or ultimo_salvo >= mais_recente_anp:
        return [mais_recente_anp]
    meses = []
    atual = proximo_mes(*ultimo_salvo)
    while atual <= mais_recente_anp:
        meses.append(atual)
        atual = proximo_mes(*atual)
    return meses


########################################################################
# BLOCO 10 — PONTO DE ENTRADA (CLI), COM BACKFILL AUTOMATICO
########################################################################
# Objetivo: amarrar os blocos anteriores; rodar sozinho (detecta o mes,
# BLOCO 4) ou forcado (--ano/--mes, util para testar um mes especifico).
#
# Racional do backfill: `detectar_mes_mais_recente` so devolve UM mes — o
# mais novo disponivel. Isso e suficiente no caso normal (a automacao
# roda todo mes, sem atraso acumulado), mas se o pipeline ficar parado
# por um tempo (ou a ANP atrasar a publicacao, como ja aconteceu), varios
# meses ficam de fora do `historico` de uma vez. Em vez de processar so o
# mes mais novo e deixar um buraco no grafico de tendencia, comparamos
# com o ultimo mes ja salvo (BLOCO 9) e processamos, em ordem, todos os
# meses que faltam ate o mais recente — no caso comum isso e so 1 mes
# (comportamento identico ao anterior), sem custo extra.
#
# --ano/--mes continuam forcando um unico mes especifico, sem backfill —
# uteis para testar/reprocessar um mes pontual sem mexer no historico.
########################################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ano", type=int, default=None)
    parser.add_argument("--mes", type=int, default=None)
    args = parser.parse_args()

    caminho_saida = RAIZ / "docs" / "dados" / "latest.json"
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    if args.ano and args.mes:
        meses_a_processar = [(args.ano, args.mes)]
    else:
        mais_recente = detectar_mes_mais_recente()  # BLOCO 4
        ultimo_salvo = ultimo_mes_no_historico(caminho_saida)  # BLOCO 9
        meses_a_processar = meses_faltantes(ultimo_salvo, mais_recente)  # BLOCO 9
        if len(meses_a_processar) > 1:
            print(f"Backfill: {len(meses_a_processar)} meses em atraso, processando um a um...")

    meses_pulados = []
    for ano, mes in meses_a_processar:
        print(f"Processando referencia {ano}-{mes:02d}...")
        try:
            resumo = rodar_pipeline(ano, mes)  # BLOCOS 5, 6, 7 e 8
        except RuntimeError as erro:
            # Um mes que nao bate com nenhum padrao conhecido de URL (ver
            # BLOCO 2) nao deve derrubar o backfill inteiro por causa de
            # 1 mes fora do padrao do lado da ANP — pula e segue os demais.
            print(f"  -> pulado ({erro})")
            meses_pulados.append(f"{ano}-{mes:02d}")
            continue
        resumo = atualizar_historico(resumo, caminho_saida)  # BLOCO 9
        caminho_saida.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {resumo['total_anomalias']} / {resumo['total_registros']} ({resumo['pct_anomalias']}%)")

    print(f"Salvo em {caminho_saida}")
    if meses_pulados:
        print(f"Meses pulados (URL fora dos padroes conhecidos): {', '.join(meses_pulados)}")


if __name__ == "__main__":
    sys.exit(main())
