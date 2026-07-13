"""Construção do grafo de conhecimento a partir das triplas extraídas.

Usamos networkx.DiGraph (grafo DIRECIONADO) porque as relações têm direção
semântica: "sintoma indica_causa causa" não é o mesmo que o inverso. A direção
é o que permite percorrer a cadeia causal na ordem certa durante a consulta.

Sem banco de grafo externo nesta versão: o grafo cabe em memória e o objetivo
é aprender a mecânica, não operar em escala.
"""

import json
from pathlib import Path

import networkx as nx

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_TRIPLAS = RAIZ / "data" / "processed" / "triplas.json"
ARQUIVO_GRAFO = RAIZ / "data" / "processed" / "grafo.graphml"


def construir_grafo(triplas: list[dict]) -> nx.DiGraph:
    """Monta o DiGraph: nó = entidade (com tipo), aresta = relação (com fonte)."""
    grafo = nx.DiGraph()
    for t in triplas:
        # add_node é idempotente: a mesma entidade citada em vários documentos
        # vira UM único nó — é assim que o conhecimento de fontes diferentes
        # se conecta, coisa que chunks isolados de um RAG nunca fazem.
        grafo.add_node(t["origem"], tipo=t["tipo_origem"])
        grafo.add_node(t["destino"], tipo=t["tipo_destino"])
        grafo.add_edge(
            t["origem"],
            t["destino"],
            tipo_relacao=t["relacao"],
            fonte=t["fonte"],
        )
    return grafo


def salvar_grafo(grafo: nx.DiGraph, caminho: Path = ARQUIVO_GRAFO) -> None:
    """Persiste em GraphML — formato texto, legível e aberto por outras
    ferramentas (Gephi, yEd), o que ajuda a inspecionar/depurar o grafo."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(grafo, caminho, encoding="utf-8")


def carregar_grafo(caminho: Path = ARQUIVO_GRAFO) -> nx.DiGraph:
    return nx.read_graphml(caminho)


def executar_construcao() -> nx.DiGraph:
    dados = json.loads(ARQUIVO_TRIPLAS.read_text(encoding="utf-8"))
    grafo = construir_grafo(dados["triplas"])
    salvar_grafo(grafo)
    print(
        f"Grafo construído: {grafo.number_of_nodes()} nós, "
        f"{grafo.number_of_edges()} arestas. Salvo em {ARQUIVO_GRAFO}"
    )
    return grafo


if __name__ == "__main__":
    executar_construcao()
