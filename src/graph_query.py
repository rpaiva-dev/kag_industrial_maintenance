"""Consulta ao grafo: identificação da entidade de partida + raciocínio multi-hop.

Aqui está o coração da diferença para um RAG comum:
- RAG: embedda a pergunta, busca chunks parecidos, devolve texto solto.
- KAG: (1) o LLM só faz o "linking" — mapear a pergunta para UM nó do grafo;
  (2) o PERCURSO é feito por algoritmo determinístico (BFS/DFS), não pelo LLM.
  Cada salto segue uma aresta que existe de fato, então a cadeia
  sintoma -> causa -> ação é garantidamente suportada pelos dados.

Além disso, o sistema pede mais informação ao usuário quando a pergunta não é
específica o suficiente para escolher um caminho — por exemplo, perguntar
sobre um COMPONENTE sem dizer o SINTOMA observado. Essa checagem também é
determinística: olhamos quantos "próximos passos" distintos existem no grafo
a partir da entidade identificada. Se há mais de um, a pergunta é ambígua e
pedimos ao usuário para escolher, em vez de adivinhar.
"""

import json

import networkx as nx

from src.llm_client import chamar_llm

# Limite de saltos: 3 cobre a cadeia sintoma -> causa -> ação (e sobra um salto
# quando a partida é um componente). Sem limite, o BFS traria o grafo inteiro
# e afogaria o prompt final em contexto irrelevante.
MAX_SALTOS = 3

SCHEMA_ENTIDADE = {
    "type": "object",
    "properties": {
        # anyOf com null: o LLM precisa poder dizer "nenhuma entidade casa" —
        # é isso que permite ao sistema admitir que não sabe, em vez de forçar
        # um match ruim e alucinar uma resposta.
        "entidade": {"anyOf": [{"type": "string"}, {"type": "null"}]}
    },
    "required": ["entidade"],
    "additionalProperties": False,
}


def identificar_entidade_partida(pergunta: str, grafo: nx.DiGraph) -> str | None:
    """Usa o LLM para mapear a pergunta a um nó existente do grafo.

    Passamos a lista completa de nós (com tipo) no prompt — viável porque o
    grafo é pequeno. O LLM escolhe dentre opções REAIS, nunca cria nomes novos;
    ainda assim validamos que a resposta é um nó de verdade.
    """
    linhas = [
        f"- {no} (tipo: {dados.get('tipo', '?')})" for no, dados in grafo.nodes(data=True)
    ]
    resposta = chamar_llm(
        system=(
            "Você mapeia perguntas sobre manutenção de ativos de mineração "
            "(peneiras, transportadores de correia, caminhões fora de estrada, "
            "britadores) para entidades de um grafo de conhecimento. Dada a "
            "pergunta e a lista de entidades, retorne o nome EXATO da entidade "
            "MAIS ESPECÍFICA mencionada na pergunta como ponto de partida do "
            "raciocínio: prefira um sintoma citado; se não houver sintoma mas "
            "houver um componente citado, retorne o componente; se só o "
            "equipamento for citado, retorne o equipamento.\n\n"
            "REGRA CRÍTICA — nunca troque de equipamento: se a pergunta "
            "menciona explicitamente um tipo de equipamento (peneira, "
            "transportador, caminhão ou britador), a entidade escolhida DEVE "
            "pertencer a ESSE equipamento. Nunca escolha uma entidade de um "
            "equipamento diferente do mencionado só porque o nome parece "
            "combinar (ex: se a pergunta é sobre o britador, jamais retorne "
            "uma entidade que só existe na peneira, mesmo que a palavra do "
            "sintoma seja parecida). Se nenhuma entidade do equipamento "
            "correto corresponder ao sintoma descrito, prefira retornar o "
            "componente ou o equipamento certos (mesmo sem sintoma "
            "correspondente) a retornar uma entidade de outro equipamento; "
            "se nem isso houver, retorne null.\n\n"
            "Se a pergunta mencionar mais de um turno de conversa (contexto "
            "anterior + nova resposta do usuário), combine as informações "
            "para achar a entidade mais específica possível. Se NENHUMA "
            "entidade da lista corresponder ao assunto da pergunta, retorne "
            "null. Nunca invente um nome fora da lista."
        ),
        user=f"Pergunta: {pergunta}\n\nEntidades do grafo:\n" + "\n".join(linhas),
        json_schema=SCHEMA_ENTIDADE,
    )
    entidade = json.loads(resposta)["entidade"]
    # Cinto e suspensório: só aceitamos se for realmente um nó do grafo.
    if entidade is not None and entidade in grafo:
        return entidade
    return None


def _sucessores_por_relacao(grafo: nx.DiGraph, no: str, relacao: str) -> list[str]:
    """Vizinhos de saída de `no` conectados especificamente por `relacao`."""
    return sorted(
        v for v in grafo.successors(no) if grafo.edges[no, v]["tipo_relacao"] == relacao
    )


def verificar_especificidade(entidade: str, grafo: nx.DiGraph) -> dict:
    """Checa, de forma determinística, se a entidade é específica o bastante.

    A ideia central: se o próximo passo da cadeia causal a partir da entidade
    tem MAIS DE UMA opção (ex: um componente com vários sintomas possíveis, ou
    um equipamento com vários componentes possíveis), a pergunta não deu
    informação suficiente para escolher qual caminho seguir — então paramos
    aqui e devolvemos as opções para o usuário escolher, em vez de deixar o
    LLM advinhar (e possivelmente citar uma causa que não é a do problema
    real do usuário).

    Sintomas com múltiplas causas, ou causas com múltiplas ações, NÃO contam
    como ambiguidade: nesse nível é normal e útil listar todas as
    possibilidades na resposta final (o answer_builder já faz isso bem).
    """
    tipo = grafo.nodes[entidade].get("tipo")

    if tipo == "equipamento":
        candidatos = _sucessores_por_relacao(grafo, entidade, "tem_componente")
        if len(candidatos) > 1:
            return {
                "precisa_esclarecimento": True,
                "entidade_ambigua": entidade,
                "relacao_ambigua": "tem_componente",
                "candidatos": candidatos,
                "pergunta_esclarecimento": (
                    f"O(a) **{entidade}** tem vários componentes que podem estar "
                    f"envolvidos: {', '.join(candidatos)}. Qual componente está "
                    f"apresentando o problema?"
                ),
            }
        if len(candidatos) == 1:
            # Só um componente possível: resolve automaticamente e desce mais
            # um nível para checar se o COMPONENTE, por sua vez, é específico.
            return verificar_especificidade(candidatos[0], grafo)
        return {"precisa_esclarecimento": False, "entidade_resolvida": entidade}

    if tipo == "componente":
        candidatos = _sucessores_por_relacao(grafo, entidade, "apresenta_sintoma")
        if len(candidatos) > 1:
            return {
                "precisa_esclarecimento": True,
                "entidade_ambigua": entidade,
                "relacao_ambigua": "apresenta_sintoma",
                "candidatos": candidatos,
                "pergunta_esclarecimento": (
                    f"O(a) **{entidade}** pode apresentar diferentes sintomas: "
                    f"{', '.join(candidatos)}. Qual desses sintomas você está "
                    f"observando?"
                ),
            }
        return {"precisa_esclarecimento": False, "entidade_resolvida": entidade}

    # sintoma, causa ou ação: já é o nível mais específico possível — nada a
    # esclarecer, segue para a travessia normal.
    return {"precisa_esclarecimento": False, "entidade_resolvida": entidade}


def coletar_caminhos(grafo: nx.DiGraph, partida: str, max_saltos: int = MAX_SALTOS) -> list[list[dict]]:
    """Percorre o grafo a partir da entidade e devolve caminhos como listas de triplas.

    Estratégia: DFS limitada em profundidade seguindo as arestas de SAÍDA
    (a direção causal: sintoma -> causa -> ação). Cada caminho retornado é uma
    sequência ordenada de triplas — exatamente a "evidência" que o LLM usará
    para responder. Também incluímos 1 salto de ENTRADA (quem aponta para a
    partida), para dar contexto: ex. qual componente apresenta o sintoma.
    """
    caminhos: list[list[dict]] = []

    def dfs(no: str, caminho_atual: list[dict], visitados: set[str]):
        sucessores = [v for v in grafo.successors(no) if v not in visitados]
        # Nó folha ou limite de saltos atingido: fecha o caminho acumulado.
        if not sucessores or len(caminho_atual) >= max_saltos:
            if caminho_atual:
                caminhos.append(list(caminho_atual))
            return
        for v in sucessores:
            aresta = grafo.edges[no, v]
            tripla = {
                "origem": no,
                "relacao": aresta["tipo_relacao"],
                "destino": v,
                "fonte": aresta.get("fonte", ""),
            }
            caminho_atual.append(tripla)
            dfs(v, caminho_atual, visitados | {v})
            caminho_atual.pop()

    dfs(partida, [], {partida})

    # Contexto reverso (1 salto): ex. "motor da peneira apresenta_sintoma vibração alta"
    # quando a partida é o sintoma. Ajuda a resposta a citar o componente afetado.
    for u in grafo.predecessors(partida):
        aresta = grafo.edges[u, partida]
        caminhos.append([
            {
                "origem": u,
                "relacao": aresta["tipo_relacao"],
                "destino": partida,
                "fonte": aresta.get("fonte", ""),
            }
        ])

    return caminhos


def _caminhos_das_opcoes(grafo: nx.DiGraph, checagem: dict) -> list[list[dict]]:
    """Monta 'caminhos' de 1 salto representando as opções candidatas.

    Usado só para exibir visualmente, no subgrafo do chat, os pontos que o
    usuário precisa escolher entre — sem que isso conte como uma resposta.
    """
    no = checagem["entidade_ambigua"]
    relacao = checagem["relacao_ambigua"]
    return [
        [
            {
                "origem": no,
                "relacao": relacao,
                "destino": c,
                "fonte": grafo.edges[no, c].get("fonte", ""),
            }
        ]
        for c in checagem["candidatos"]
    ]


def consultar(pergunta: str, grafo: nx.DiGraph, contexto_anterior: str | None = None) -> dict:
    """Orquestra a consulta: linking + checagem de especificidade + travessia.

    contexto_anterior: quando a resposta anterior do assistente foi um pedido
    de esclarecimento, passamos a pergunta original do usuário junto com a
    nova resposta dele, para que o LLM identifique a entidade combinando as
    duas informações (ex: "motor do britador" + "vibração alta").
    """
    pergunta_efetiva = f"{contexto_anterior}\n{pergunta}" if contexto_anterior else pergunta
    partida = identificar_entidade_partida(pergunta_efetiva, grafo)

    if partida is None:
        return {
            "entidade_partida": None,
            "caminhos": [],
            "precisa_esclarecimento": False,
            "pergunta_esclarecimento": None,
        }

    checagem = verificar_especificidade(partida, grafo)

    if checagem["precisa_esclarecimento"]:
        return {
            "entidade_partida": checagem["entidade_ambigua"],
            "caminhos": _caminhos_das_opcoes(grafo, checagem),
            "precisa_esclarecimento": True,
            "pergunta_esclarecimento": checagem["pergunta_esclarecimento"],
        }

    entidade_resolvida = checagem["entidade_resolvida"]
    return {
        "entidade_partida": entidade_resolvida,
        "caminhos": coletar_caminhos(grafo, entidade_resolvida),
        "precisa_esclarecimento": False,
        "pergunta_esclarecimento": None,
    }
