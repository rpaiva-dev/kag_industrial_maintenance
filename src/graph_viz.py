"""Visualização do grafo com destaque do caminho usado na última resposta.

Usamos pyvis (rede interativa em HTML) em vez de matplotlib estático porque,
para portfólio, poder arrastar/zoom e VER o raciocínio destacado comunica
muito mais do que uma imagem congelada. O HTML gerado é embutido no Streamlit
via st.components.v1.html.

Este módulo também gera a ANIMAÇÃO DE RASTREIO: enquanto o agente está
respondendo, mostramos o subgrafo se "construindo" nó a nó, na ordem em que a
travessia (BFS a partir da entidade de partida) os alcança — simulando
visualmente o algoritmo de graph_query.py explorando a base. Isso usa uma
página HTML/JS minimalista própria (não o template do pyvis), porque
precisamos adicionar nós/arestas ao vis-network progressivamente via
setTimeout, e o pyvis só gera datasets estáticos completos.
"""

import json
from collections import deque

import networkx as nx
from pyvis.network import Network

# Cores por tipo de entidade — a legenda visual da cadeia causal.
CORES_TIPO = {
    "equipamento": "#4e79a7",
    "componente": "#59a14f",
    "sintoma": "#f28e2b",
    "causa": "#e15759",
    "acao": "#b07aa1",
}
COR_PADRAO = "#9c9c9c"
COR_DESTAQUE = "#ffd700"  # dourado: nós/arestas do caminho da resposta


def subgrafo_dos_caminhos(grafo: nx.DiGraph, caminhos: list[list[dict]]) -> nx.DiGraph:
    """Constrói um grafo contendo APENAS os nós/arestas percorridos na resposta.

    Por que um subgrafo e não o grafo inteiro com destaque? Para o usuário
    final, o que importa é ver os pontos que a consulta conectou na base —
    mostrar as ~100 entidades todas vira ruído visual. O subgrafo é a
    "trilha de raciocínio" isolada.
    """
    sub = nx.DiGraph()
    for caminho in caminhos:
        for t in caminho:
            for no in (t["origem"], t["destino"]):
                # Copia o atributo tipo do grafo original para manter as cores.
                sub.add_node(no, tipo=grafo.nodes[no].get("tipo", "?") if no in grafo else "?")
            sub.add_edge(t["origem"], t["destino"], tipo_relacao=t["relacao"], fonte=t.get("fonte", ""))
    return sub


def gerar_html_grafo(grafo: nx.DiGraph, caminhos: list[list[dict]] | None = None) -> str:
    """Gera o HTML da rede, destacando os caminhos (se fornecidos)."""
    # Conjuntos de nós/arestas a destacar, derivados das triplas dos caminhos.
    nos_destaque: set[str] = set()
    arestas_destaque: set[tuple[str, str]] = set()
    for caminho in caminhos or []:
        for t in caminho:
            nos_destaque.update([t["origem"], t["destino"]])
            arestas_destaque.add((t["origem"], t["destino"]))

    rede = Network(height="550px", width="100%", directed=True, bgcolor="#ffffff")
    rede.barnes_hut()  # física padrão razoável para grafos pequenos

    for no, dados in grafo.nodes(data=True):
        tipo = dados.get("tipo", "?")
        destacado = no in nos_destaque
        rede.add_node(
            no,
            label=no,
            title=f"{no} ({tipo})",
            color=CORES_TIPO.get(tipo, COR_PADRAO),
            # Borda dourada + tamanho maior sinalizam o caminho do raciocínio
            # sem perder a cor do tipo (que carrega a semântica da cadeia).
            borderWidth=4 if destacado else 1,
            size=28 if destacado else 16,
            **({"borderWidthSelected": 4, "shapeProperties": {"borderDashes": False}} if destacado else {}),
        )
        if destacado:
            rede.get_node(no)["color"] = {
                "background": CORES_TIPO.get(tipo, COR_PADRAO),
                "border": COR_DESTAQUE,
            }

    for u, v, dados in grafo.edges(data=True):
        destacada = (u, v) in arestas_destaque
        rede.add_edge(
            u,
            v,
            label=dados.get("tipo_relacao", ""),
            color=COR_DESTAQUE if destacada else "#c0c0c0",
            width=4 if destacada else 1,
            font={"size": 10, "align": "middle"},
        )

    # generate_html devolve a página completa como string — evitamos escrever
    # arquivo temporário em disco só para reler em seguida.
    return rede.generate_html(notebook=False)


def gerar_passos_rastreio(
    subgrafo: nx.DiGraph, raiz: str, max_nos: int = 8
) -> list[dict]:
    """Gera a sequência ordenada de passos (nó, aresta, nó, aresta...) que a
    animação vai revelar, explorando o subgrafo em BFS a partir da raiz.

    BFS (mais próximo primeiro) em vez da ordem bruta das triplas porque é
    assim que uma exploração "de dentro para fora" a partir da entidade de
    partida se pareceria de verdade — e reaproveita arestas de entrada e
    saída (contexto reverso + cadeia causal) na ordem em que seriam
    descobertas a partir da raiz, não na ordem em que aparecem nas triplas.
    """
    if raiz not in subgrafo:
        return []

    visitados = {raiz}
    fila = deque([raiz])
    passos = [
        {
            "tipo": "no",
            "id": raiz,
            "label": raiz,
            "cor": CORES_TIPO.get(subgrafo.nodes[raiz].get("tipo"), COR_PADRAO),
        }
    ]

    while fila and len(visitados) < max_nos:
        atual = fila.popleft()
        vizinhos = list(subgrafo.successors(atual)) + list(subgrafo.predecessors(atual))
        for v in vizinhos:
            if v in visitados or len(visitados) >= max_nos:
                continue
            visitados.add(v)
            fila.append(v)
            # A aresta pode existir em qualquer direção entre atual e v.
            de, para = (atual, v) if subgrafo.has_edge(atual, v) else (v, atual)
            passos.append(
                {
                    "tipo": "aresta",
                    "id": f"{de}=>{para}",
                    "de": de,
                    "para": para,
                    "label": subgrafo.edges[de, para]["tipo_relacao"],
                }
            )
            passos.append(
                {
                    "tipo": "no",
                    "id": v,
                    "label": v,
                    "cor": CORES_TIPO.get(subgrafo.nodes[v].get("tipo"), COR_PADRAO),
                }
            )

    return passos


def gerar_html_rastreio(
    grafo: nx.DiGraph,
    caminhos: list[list[dict]],
    raiz: str,
    mensagem_final: str = "Caminho encontrado.",
    duracao_passo_ms: int = 450,
    max_nos: int = 8,
) -> tuple[str, int]:
    """Gera a animação de rastreio do grafo e devolve (html, nº de passos).

    O nº de passos é devolvido para quem chama poder estimar quanto tempo a
    animação leva (e, se necessário, aguardar esse tempo antes de trocá-la
    pela resposta final — ver app.py).
    """
    sub = subgrafo_dos_caminhos(grafo, caminhos)
    passos = gerar_passos_rastreio(sub, raiz, max_nos=max_nos)
    passos_json = json.dumps(passos, ensure_ascii=False)
    duracao_total_ms = len(passos) * duracao_passo_ms

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; font-family: -apple-system, Segoe UI, sans-serif; background: #ffffff; }}
  #rede {{ width: 100%; height: 400px; }}
  #status {{
    text-align: center; font-size: 13px; color: #888; padding: 8px 0 2px 0;
    transition: color .3s ease;
  }}
  #status.pronto {{ color: #2a9d5c; font-weight: 600; }}
</style>
</head>
<body>
<div id="rede"></div>
<div id="status">🔎 Percorrendo o grafo de conhecimento...</div>
<script>
  const passos = {passos_json};
  const DELAY = {duracao_passo_ms};

  const nodes = new vis.DataSet([]);
  const edges = new vis.DataSet([]);
  const container = document.getElementById('rede');
  const network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, {{
    physics: {{ stabilization: {{ iterations: 80 }}, barnesHut: {{ springLength: 120 }} }},
    layout: {{ improvedLayout: true }},
    interaction: {{ dragNodes: true, zoomView: true }},
    edges: {{ arrows: 'to', font: {{ size: 11, align: 'middle' }}, smooth: {{ type: 'continuous' }} }},
    nodes: {{ shape: 'dot', font: {{ size: 13 }} }},
  }});

  passos.forEach((passo, i) => {{
    setTimeout(() => {{
      if (passo.tipo === 'no') {{
        // Flash dourado ao "descobrir" o nó, depois assenta na cor do tipo —
        // visualmente comunica "acabei de chegar aqui" -> "nó conhecido".
        nodes.add({{ id: passo.id, label: passo.label, color: '#ffd700', borderWidth: 4, size: 24 }});
        setTimeout(() => {{
          try {{ nodes.update({{ id: passo.id, color: passo.cor, borderWidth: 2, size: 16 }}); }} catch (e) {{}}
        }}, Math.max(DELAY - 120, 100));
      }} else {{
        edges.add({{ id: passo.id, from: passo.de, to: passo.para, label: passo.label, color: {{ color: '#ffd700' }}, width: 3 }});
      }}
    }}, i * DELAY);
  }});

  setTimeout(() => {{
    const st = document.getElementById('status');
    if (st) {{
      st.textContent = {json.dumps("✅ " + mensagem_final)};
      st.classList.add('pronto');
    }}
  }}, Math.max(passos.length, 1) * DELAY + 150);
</script>
</body>
</html>"""

    return html, len(passos)
