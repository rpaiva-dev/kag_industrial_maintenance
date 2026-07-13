# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projeto: KAG — Manutenção Industrial

Sistema de KAG (Knowledge Augmented Generation) construído do ZERO, sem frameworks prontos de grafo de conhecimento (nada de LangChain GraphRAG, LlamaIndex KG, etc). O objetivo é aprender extração de entidades/relações, construção de grafo e raciocínio multi-hop, implementando cada peça manualmente.

Domínio: manutenção de ativos de **mineração** — atualmente 4 tipos de ativo: peneira vibratória, transportador de correia, caminhão fora de estrada e britador. A base cobre equipamentos, componentes (motores, molas, mancais, etc.), sintomas de falha, causas prováveis e ações corretivas. Diferente de RAG puro (busca por similaridade de texto), o sistema responde perguntas que exigem atravessar uma CADEIA de relações causa-efeito: Equipamento → Componente → Sintoma → Causa → Ação corretiva.

Quando a pergunta do usuário não é específica o suficiente para escolher um caminho (ex: cita um componente mas não o sintoma, ou o equipamento mas não o componente), o agente PEDE mais detalhes em vez de adivinhar — essa checagem é determinística, baseada na estrutura do grafo (`src/graph_query.py::verificar_especificidade`), não uma decisão do LLM.

## Comandos

```powershell
# Setup (uma vez)
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip -q
.venv/Scripts/python -m pip install -r requirements.txt -q

# Pipeline offline: extrair triplas dos documentos e construir o grafo
.venv/Scripts/python -m src.extraction     # data/raw -> data/processed/triplas.json
.venv/Scripts/python -m src.graph_builder  # triplas.json -> data/processed/grafo.graphml

# Rodar o app
.venv/Scripts/python -m streamlit run app.py --server.headless true
```

Sempre use o interpretador do `.venv` (`.venv/Scripts/python` no Windows), nunca o Python do sistema.

A chave da API vai em `.env` local (`ANTHROPIC_API_KEY=...`, ver `.env.example`) — nunca hardcoded. Em produção Streamlit, usar `st.secrets`.

## Arquitetura

Pipeline em duas fases:

**Fase offline (indexação):**
1. `data/raw/` — documentos sintéticos de manuais de manutenção (JSON com metadado de origem)
2. `src/extraction.py` — LLM (Claude via lib `anthropic`) extrai triplas `(origem, relação, destino)` com tipos de entidade; relações permitidas: `tem_componente`, `apresenta_sintoma`, `indica_causa`, `resolve_com`. Saída validada em `data/processed/triplas.json`
3. `src/graph_builder.py` — monta `nx.DiGraph` (nós com atributo `tipo`, arestas com `tipo_relacao` e `fonte`) e persiste em `.graphml`

**Fase online (consulta, orquestrada pelo `app.py`):**
4. `src/graph_query.py` — LLM identifica a entidade de partida na pergunta; BFS limitada (até 3 saltos) coleta caminhos relevantes como listas de triplas ordenadas
5. `src/answer_builder.py` — prompt final com os caminhos do grafo + instrução de responder SOMENTE com base nas relações do grafo, explicitando o caminho. Se não há entidade de partida, o sistema admite não ter a informação — nunca inventa relação
6. `src/graph_viz.py` — visualização pyvis do grafo destacando o caminho usado na última resposta
7. `app.py` — front-end Streamlit: campo de pergunta, resposta + caminho em texto + grafo, histórico em `st.session_state`

`src/llm_client.py` centraliza toda chamada à API Anthropic (modelo `claude-opus-4-8`).

## Convenções

- Comentários de código em português, explicando o "porquê" das decisões técnicas
- Sem frameworks de KG — apenas `anthropic`, `networkx`, `pyvis`, `streamlit`
- O arquivo `index.html` só deve ser criado/alterado via skill `format_index_to_landing_page` (comando `/format_index_to_landing_page`), e somente quando pedido explicitamente
