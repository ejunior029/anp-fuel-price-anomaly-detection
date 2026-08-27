"""Pipeline de scoring mensal: baixa o CSV mais recente da ANP, aplica o
modelo ja treinado (sem retreinar) e gera o JSON consumido pelo mapa em
docs/index.html.

Uso:
    python src/pipeline_mensal.py                  # tenta detectar o mes mais recente
    python src/pipeline_mensal.py --ano 2025 --mes 12   # forca um mes especifico

Pensado para rodar tanto localmente quanto no GitHub Actions
(.github/workflows/atualizar_mapa.yml), reaproveitando os artefatos de
producao salvos em models/ pelo notebook 04 (referencia_precos.joblib,
preprocessador.joblib, melhor_modelo.joblib).
"""
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

# www.gov.br tem endereco IPv6, e runners do GitHub Actions costumam nao ter
# rota IPv6 funcional para a internet ("Network is unreachable" mesmo com
# IPv4 acessivel). Forcamos urllib3 (usado por `requests`) a so tentar IPv4.
_urllib3_conn.allowed_gai_family = lambda: socket.AF_INET

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))  # permite `from src.xxx import ...` mesmo rodando este arquivo direto

URL_BASE = (
    "https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/"
    "arquivos/shpc/dsan/{ano}/precos-gasolina-etanol-{mes:02d}.csv"
)

# Nomes proprios oficiais dos estados, com acentuacao correta — sao exibidos
# diretamente na pagina publica (docs/index.html), inclusive na versao em
# ingles: nomes proprios brasileiros nao sao traduzidos, so mantidos com o
# acento certo (e assim que aparecem em textos em ingles tambem).
NOME_ESTADO = {
    "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá",
    "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo",
    "GO": "Goiás", "MA": "Maranhão", "MG": "Minas Gerais", "MS": "Mato Grosso do Sul",
    "MT": "Mato Grosso", "PA": "Pará", "PB": "Paraíba", "PE": "Pernambuco",
    "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul", "SC": "Santa Catarina",
    "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins",
}


def baixar_csv(ano, mes, destino):
    """Baixa o CSV da ANP para `destino`. Devolve True se conseguiu."""
    url = URL_BASE.format(ano=ano, mes=mes)
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    if resp.status_code != 200 or len(resp.content) < 1000:
        return False
    destino.write_bytes(resp.content)
    return True


def detectar_mes_mais_recente(max_tentativas=14):
    """Tenta o mes atual e volta ate `max_tentativas` meses se o arquivo
    ainda nao tiver sido publicado pela ANP.

    A ANP as vezes fica meses em atraso na publicacao (ja observamos um
    atraso de 8 meses em producao), entao a janela de busca precisa ser
    generosa — bem maior do que os 2-3 meses que pareceriam suficientes
    em um cenario "normal"."""
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


def carregar_dados_bruto(caminho):
    """Mesma limpeza do src.data.carregar_dados, mas a partir de um
    caminho ja baixado (evita duplicar a funcao)."""
    from src.data import carregar_dados
    return carregar_dados(caminho)


def rodar_pipeline(ano, mes):
    from src.evaluation import calcular_metricas  # noqa: F401 (mantido para simetria com os notebooks)
    from src.models import prever_anomalias

    caminho_csv = RAIZ / "data" / f"precos_combustiveis_anp_{ano}-{mes:02d}.csv"
    caminho_csv.parent.mkdir(exist_ok=True)
    if not caminho_csv.exists():
        ok = baixar_csv(ano, mes, caminho_csv)
        if not ok:
            raise RuntimeError(f"Nao foi possivel baixar o CSV de {ano}-{mes:02d}.")

    df = carregar_dados_bruto(caminho_csv)

    referencia = joblib.load(RAIZ / "models" / "referencia_precos.joblib")
    preprocessador = joblib.load(RAIZ / "models" / "preprocessador.joblib")
    modelo = joblib.load(RAIZ / "models" / "melhor_modelo.joblib")

    df_feat = referencia.transform(df)
    X = preprocessador.transform(df_feat)
    rotulo, score = prever_anomalias(modelo, X)

    df_feat = df_feat.copy()
    df_feat["anomalia"] = rotulo
    df_feat["score_anomalia"] = score

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

    top_nacional = (
        df_feat.sort_values("score_anomalia", ascending=False)
        # o mesmo posto pode ser pesquisado varias vezes no mes; sem isso o
        # top nacional fica repetindo o mesmo caso em vez de mostrar 15 diferentes
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ano", type=int, default=None)
    parser.add_argument("--mes", type=int, default=None)
    args = parser.parse_args()

    if args.ano and args.mes:
        ano, mes = args.ano, args.mes
    else:
        ano, mes = detectar_mes_mais_recente()

    print(f"Processando referencia {ano}-{mes:02d}...")
    resumo = rodar_pipeline(ano, mes)

    caminho_saida = RAIZ / "docs" / "dados" / "latest.json"
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    resumo = atualizar_historico(resumo, caminho_saida)

    caminho_saida.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvo em {caminho_saida}")
    print(f"Total de anomalias: {resumo['total_anomalias']} / {resumo['total_registros']} ({resumo['pct_anomalias']}%)")


if __name__ == "__main__":
    sys.exit(main())
