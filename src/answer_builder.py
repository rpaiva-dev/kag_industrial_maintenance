"""Geração da resposta final a partir dos caminhos encontrados no grafo.

O prompt final é deliberadamente RESTRITIVO: o LLM só pode usar as relações
fornecidas (as triplas dos caminhos), nunca conhecimento próprio. É o inverso
de um RAG comum, onde o LLM recebe texto solto e pode "completar" com o que
sabe. Aqui, se o grafo não tem a informação, a resposta é "não sei" — a
groundedness vem da estrutura, não da boa vontade do modelo.
"""

from src.llm_client import chamar_llm

MENSAGEM_SEM_INFORMACAO = (
    "Não encontrei essa informação na base de conhecimento. "
    "O grafo não possui nenhuma entidade relacionada à sua pergunta, "
    "então prefiro admitir que não sei a inventar uma resposta."
)

PROMPT_SISTEMA = """Você é um assistente de manutenção industrial que responde EXCLUSIVAMENTE com base em um grafo de conhecimento.

Você receberá a pergunta do usuário e os caminhos percorridos no grafo (sequências de triplas origem -relação-> destino, com a fonte documental de cada relação).

Regras OBRIGATÓRIAS:
1. Use SOMENTE as relações fornecidas. Não acrescente causas, ações ou fatos que não estejam nas triplas, mesmo que você os conheça.
2. Explicite na resposta o CAMINHO percorrido no grafo que sustenta a conclusão, no formato: entidade -[relação]-> entidade -[relação]-> entidade.
3. Se os caminhos fornecidos não forem suficientes para responder, diga claramente que a base de conhecimento não cobre esse ponto.
4. Cite a fonte documental (campo "fonte") das relações usadas.
5. Responda em português, de forma clara e direta: causa(s) provável(is) primeiro, depois a(s) ação(ões) corretiva(s)."""


def formatar_caminhos(caminhos: list[list[dict]]) -> str:
    """Serializa os caminhos em texto legível para o prompt.

    Formato "a -[rel]-> b -[rel]-> c" em vez de JSON cru: mais compacto e o
    modelo replica esse formato na resposta ao explicitar o caminho.
    """
    linhas = []
    for i, caminho in enumerate(caminhos, 1):
        passos = " ".join(
            f"{t['origem']} -[{t['relacao']}]-> {t['destino']}" if j == 0
            else f"-[{t['relacao']}]-> {t['destino']}"
            for j, t in enumerate(caminho)
        )
        fontes = sorted({t["fonte"] for t in caminho if t.get("fonte")})
        linhas.append(f"Caminho {i}: {passos}\n  Fontes: {'; '.join(fontes)}")
    return "\n".join(linhas)


def gerar_resposta(pergunta: str, resultado_consulta: dict) -> str:
    """Monta o prompt final e gera a resposta ancorada no grafo."""
    if resultado_consulta["entidade_partida"] is None:
        # Curto-circuito SEM chamar o LLM: se não há entidade de partida,
        # não existe evidência nenhuma — qualquer resposta seria invenção.
        return MENSAGEM_SEM_INFORMACAO

    caminhos_texto = formatar_caminhos(resultado_consulta["caminhos"])
    user = (
        f"Pergunta do usuário: {pergunta}\n\n"
        f"Entidade de partida no grafo: {resultado_consulta['entidade_partida']}\n\n"
        f"Caminhos encontrados no grafo:\n{caminhos_texto}"
    )
    return chamar_llm(system=PROMPT_SISTEMA, user=user)
