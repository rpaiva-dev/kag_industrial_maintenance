# KAG — Manutenção Industrial

Sistema de **Knowledge Augmented Generation** construído do zero (sem LangChain GraphRAG, LlamaIndex KG ou similares) sobre uma base de conhecimento de manutenção industrial: equipamentos, componentes, sintomas, causas e ações corretivas.

Diferente de um RAG comum (que busca trechos de texto parecidos), o KAG responde perguntas que exigem atravessar uma **cadeia de relações causa-efeito** que não está inteira em um único trecho:

> "O motor da correia transportadora está com vibração alta — qual a causa mais provável e qual a ação corretiva?"
>
> Exige percorrer: Equipamento → Componente → Sintoma → Causa → Ação corretiva.

## Arquitetura

```
data/raw/  ──►  src/extraction.py  ──►  data/processed/triplas.json
                (LLM extrai triplas)          │
                                              ▼
                                    src/graph_builder.py  ──►  grafo.graphml (nx.DiGraph)
                                              │
Pergunta ──► src/graph_query.py ──► caminhos (BFS ≤ 3 saltos)
             (LLM só identifica               │
              a entidade de partida)          ▼
                                    src/answer_builder.py ──► resposta ancorada no grafo
                                              │
                                    src/graph_viz.py ──► grafo pyvis com caminho destacado
                                              │
                                          app.py (Streamlit)
```

Relações do grafo: `tem_componente`, `apresenta_sintoma`, `indica_causa`, `resolve_com`.

Se a pergunta não mapear para nenhuma entidade do grafo, o sistema **admite que não tem a informação** — nunca inventa uma relação.

## Como rodar

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -q

# configure a chave
copy .env.example .env   # e edite com sua ANTHROPIC_API_KEY

# pipeline offline (usa o LLM para extrair as triplas)
.venv/Scripts/python -m src.extraction
.venv/Scripts/python -m src.graph_builder

# app
.venv/Scripts/python -m streamlit run app.py
```

## Perguntas de exemplo

- Multi-hop: *"O motor da correia transportadora está com vibração alta — qual a causa provável e a ação corretiva?"*
- 1 salto: *"Quais componentes tem a bomba centrífuga?"*
- Fora do escopo: *"Como trocar o pneu de uma empilhadeira?"* → o sistema responde que não tem essa informação.
