"""Front-end Streamlit do KAG Agent AH (Asset-Health).

Layout no estilo das LLMs (ChatGPT/Claude):
- Barra lateral com o "logo" KAG Agent AH, botão de nova conversa e a lista
  de conversas da sessão (cada conversa tem seu próprio histórico).
- Menu Config com a opção "Sobre": explica o passo a passo de como a KAG
  funciona, lista os ativos monitorados (com descrição e componentes) e
  exibe o grafo de conhecimento COMPLETO.
- No chat, cada resposta mostra apenas o SUBGRAFO dos pontos que a consulta
  conectou na base — a trilha de raciocínio, sem o ruído do grafo inteiro.
- Quando a pergunta não é específica o suficiente (ex: cita um componente
  mas não o sintoma), o agente PEDE mais detalhes em vez de adivinhar, e a
  próxima mensagem do usuário é interpretada em conjunto com a pergunta
  original (contexto de esclarecimento).

Fluxo de cada pergunta:
1. graph_query.consultar        -> LLM identifica a entidade de partida;
   checagem determinística de especificidade; travessia (até 3 saltos).
2. Se precisar de esclarecimento -> exibe a pergunta e as opções, sem chamar
   o LLM de resposta (não há "evidência" suficiente ainda para responder).
3. Caso contrário, answer_builder.gerar_resposta -> LLM responde SOMENTE
   com base nos caminhos encontrados.
"""

import json
import time
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.answer_builder import gerar_resposta
from src.graph_builder import ARQUIVO_GRAFO, carregar_grafo, executar_construcao
from src.graph_query import consultar
from src.graph_viz import gerar_html_grafo, gerar_html_rastreio, subgrafo_dos_caminhos

# Tempo (s) de cada passo da animação de rastreio, espelhando o valor padrão
# usado dentro de gerar_html_rastreio — precisamos dele aqui para estimar
# quanto tempo esperar antes de trocar a animação pela resposta final.
DURACAO_PASSO_ANIMACAO_S = 0.45

st.set_page_config(page_title="KAG Agent AH", page_icon="🛠️", layout="wide")

ARQUIVO_RAW = Path(__file__).resolve().parent / "data" / "raw" / "manuais_manutencao.json"


@st.cache_resource
def obter_grafo():
    """Carrega o grafo do disco uma única vez por sessão do servidor.

    cache_resource (e não cache_data) porque o DiGraph é um objeto mutável
    compartilhado, não um dado serializável a ser copiado a cada rerun.
    """
    if not ARQUIVO_GRAFO.exists():
        from src.graph_builder import ARQUIVO_TRIPLAS

        if ARQUIVO_TRIPLAS.exists():
            return executar_construcao()
        st.error(
            "Grafo não encontrado. Rode o pipeline offline primeiro:\n\n"
            "`python -m src.extraction` e depois `python -m src.graph_builder`"
        )
        st.stop()
    return carregar_grafo()


@st.cache_data
def obter_ativos_monitorados() -> list[dict]:
    """Lê os metadados de exibição (descrição + componentes) direto do raw.

    Não usamos os nós do grafo para montar essa lista porque o nome exato que
    o LLM extrai pode variar levemente do texto de origem — os metadados aqui
    são a fonte confiável e estável para a descrição das máquinas.
    """
    dados = json.loads(ARQUIVO_RAW.read_text(encoding="utf-8"))
    return [
        {
            "fonte": doc["fonte"],
            "categoria": doc.get("categoria", ""),
            "descricao": doc.get("descricao_maquina", ""),
            "componentes": doc.get("componentes_principais", []),
        }
        for doc in dados["documentos"]
    ]


grafo = obter_grafo()

# ---------------------------------------------------------------------------
# Estado da sessão: múltiplas conversas, como nas interfaces de LLM.
# Estrutura: {id: {"titulo": str, "mensagens": [{pergunta, resposta, caminhos, esclarecimento}]}}
# ---------------------------------------------------------------------------
if "conversas" not in st.session_state:
    st.session_state.conversas = {}
if "conversa_ativa" not in st.session_state:
    st.session_state.conversa_ativa = None
if "pagina" not in st.session_state:
    st.session_state.pagina = "chat"  # "chat" | "sobre"


def nova_conversa():
    """Cria uma conversa vazia e a torna ativa (equivalente ao 'New chat')."""
    cid = str(uuid.uuid4())[:8]
    st.session_state.conversas[cid] = {"titulo": "Nova conversa", "mensagens": []}
    st.session_state.conversa_ativa = cid
    st.session_state.pagina = "chat"


# Garante que sempre existe pelo menos uma conversa aberta.
if not st.session_state.conversas:
    nova_conversa()

# ---------------------------------------------------------------------------
# BARRA LATERAL — logo, nova conversa, lista de conversas e Config.
# ---------------------------------------------------------------------------
with st.sidebar:
    # "Logo" no canto superior esquerdo, como nas LLMs.
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;padding:4px 0 12px 0;">
          <div style="font-size:30px;">🛠️</div>
          <div>
            <div style="font-size:19px;font-weight:700;line-height:1.1;">KAG Agent AH</div>
            <div style="font-size:12px;opacity:.65;">Asset-Health · Knowledge Augmented Generation</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("➕ Nova conversa", use_container_width=True):
        nova_conversa()
        st.rerun()

    st.markdown("**Conversas**")
    # Lista das conversas (mais recentes primeiro), clicáveis como nas LLMs.
    # Cada uma tem um botão de três pontos (⋮) com as opções de renomear e
    # apagar — o popover funciona como o menu de contexto.
    for cid, conv in reversed(list(st.session_state.conversas.items())):
        ativa = cid == st.session_state.conversa_ativa and st.session_state.pagina == "chat"
        rotulo = ("🟢 " if ativa else "") + conv["titulo"][:32]

        col_nome, col_menu = st.columns([5, 1])
        with col_nome:
            if st.button(rotulo, key=f"abrir_{cid}", use_container_width=True):
                st.session_state.conversa_ativa = cid
                st.session_state.pagina = "chat"
                st.rerun()
        with col_menu:
            with st.popover("⋮", use_container_width=True, key=f"menu_{cid}"):
                novo_titulo = st.text_input(
                    "Renomear conversa", value=conv["titulo"], key=f"input_renomear_{cid}"
                )
                if st.button("✏️ Renomear", key=f"btn_renomear_{cid}", use_container_width=True):
                    titulo = novo_titulo.strip()
                    if titulo:
                        conv["titulo"] = titulo
                    st.rerun()
                if st.button("🗑️ Apagar", key=f"btn_apagar_{cid}", use_container_width=True):
                    del st.session_state.conversas[cid]
                    # Se era a conversa ativa, escolhe outra existente (ou
                    # deixa None — o guard logo acima cria uma nova vazia).
                    if st.session_state.conversa_ativa == cid:
                        restantes = list(st.session_state.conversas.keys())
                        st.session_state.conversa_ativa = restantes[-1] if restantes else None
                    st.rerun()

    st.divider()

    # Menu Config: hoje só tem "Sobre", mas o popover deixa espaço para
    # futuras opções (modelo, nº de saltos, etc.) sem mudar o layout.
    with st.popover("⚙️ Config", use_container_width=True):
        if st.button("ℹ️ Sobre — como a aplicação funciona", use_container_width=True):
            st.session_state.pagina = "sobre"
            st.rerun()

# ---------------------------------------------------------------------------
# PÁGINA "SOBRE" — o passo a passo da KAG, os ativos monitorados e o grafo completo.
# ---------------------------------------------------------------------------
if st.session_state.pagina == "sobre":
    col_titulo, col_voltar = st.columns([5, 1])
    with col_titulo:
        st.title("ℹ️ Sobre — como esta KAG funciona")
    with col_voltar:
        if st.button("← Voltar ao chat"):
            st.session_state.pagina = "chat"
            st.rerun()

    st.markdown(
        """
Uma **KAG (Knowledge Augmented Generation)** ancora as respostas do LLM em um
**grafo de conhecimento**, em vez de trechos de texto soltos (como no RAG).
Isso permite responder perguntas que exigem **encadear relações** — e provar
o caminho usado, ou pedir mais detalhes quando a pergunta não é específica
o suficiente.

### Fase offline — construção da base (roda uma vez)

| Passo | O que acontece | Módulo |
|---|---|---|
| **1. Documentos** | Manuais de manutenção de 4 tipos de ativos de mineração ficam em `data/raw/`, com metadado de fonte | `data/raw/` |
| **2. Extração de triplas** | O LLM lê cada manual e extrai relações estruturadas `(origem, relação, destino)` com tipos. Só 4 relações são permitidas: `tem_componente`, `apresenta_sintoma`, `indica_causa`, `resolve_com`. Cada tripla é validada (a relação precisa conectar os tipos certos) | `src/extraction.py` |
| **3. Montagem do grafo** | As triplas viram um grafo direcionado: cada entidade é um **nó** (com tipo), cada relação é uma **aresta** (com a fonte documental). Entidades iguais citadas em manuais diferentes viram o MESMO nó | `src/graph_builder.py` |

### Fase online — o que acontece a cada pergunta sua

| Passo | O que acontece | Módulo |
|---|---|---|
| **4. Identificação da entidade** | O LLM mapeia sua pergunta para **um nó real do grafo** (ex: "vibração alta"), preferindo o mais específico possível. Se nenhum nó corresponde, o sistema **admite que não sabe** — nunca inventa | `src/graph_query.py` |
| **5. Checagem de especificidade** | Se a entidade identificada é um **componente** ou **equipamento** com mais de um próximo passo possível na cadeia (várias sintomas, ou vários componentes), o agente **pede mais detalhes** em vez de adivinhar — exatamente como um técnico faria | `src/graph_query.py` |
| **6. Travessia multi-hop** | Com uma entidade específica, um algoritmo **determinístico** (não o LLM!) percorre o grafo até 3 saltos seguindo a cadeia causal: sintoma → causa → ação | `src/graph_query.py` |
| **7. Resposta ancorada** | O LLM recebe SOMENTE os caminhos encontrados e é obrigado a responder com base neles, explicitando o caminho e citando as fontes | `src/answer_builder.py` |
| **8. Visualização** | O chat exibe o subgrafo dos pontos que foram conectados — a trilha de raciocínio, ou as opções candidatas quando o agente pede esclarecimento | `src/graph_viz.py` |

**A diferença-chave para o RAG:** no RAG o LLM recebe texto parecido e pode
"completar" com o que sabe. Na KAG, a *evidência* é estrutural: se a aresta
não existe no grafo, a relação não entra na resposta — e se falta uma
informação para escolher entre arestas, o agente pergunta.

### Cadeia causal do domínio

`Equipamento` →`tem_componente`→ `Componente` →`apresenta_sintoma`→ `Sintoma` →`indica_causa`→ `Causa` →`resolve_com`→ `Ação corretiva`

### Exemplo do fluxo de esclarecimento

> **Você:** "O motor do britador está com problema."
>
> **Agente:** "O motor de acionamento do britador pode apresentar diferentes
> sintomas: sobrecorrente no motor de acionamento do britador. Qual desses
> sintomas você está observando?" *(ou, se o componente citado tiver mais de
> um sintoma cadastrado, todas as opções aparecem para você escolher)*
>
> **Você:** "sobrecorrente"
>
> **Agente:** responde com a causa e a ação corretiva, citando a fonte.
        """
    )

    st.divider()
    st.subheader("🏭 Ativos monitorados")
    st.caption("Os 4 tipos de ativos de mineração cobertos pela base de conhecimento atual.")
    for ativo in obter_ativos_monitorados():
        with st.container(border=True):
            st.markdown(f"**{ativo['fonte']}**  \n*{ativo['categoria']}*")
            st.write(ativo["descricao"])
            st.markdown(
                "Componentes principais: "
                + ", ".join(f"`{c}`" for c in ativo["componentes"])
            )

    st.divider()
    st.subheader("Grafo de conhecimento completo")
    st.caption(
        f"{grafo.number_of_nodes()} entidades · {grafo.number_of_edges()} relações — "
        "🔵 equipamento · 🟢 componente · 🟠 sintoma · 🔴 causa · 🟣 ação"
    )
    components.html(gerar_html_grafo(grafo), height=570, scrolling=False)
    st.stop()

# ---------------------------------------------------------------------------
# PÁGINA DE CHAT
# ---------------------------------------------------------------------------
conversa = st.session_state.conversas[st.session_state.conversa_ativa]

if not conversa["mensagens"]:
    # Tela de boas-vindas de conversa vazia, como nas LLMs.
    st.markdown(
        """
        <div style="text-align:center;padding:60px 0 20px 0;">
          <div style="font-size:52px;">🛠️</div>
          <h2 style="margin:4px 0;">KAG Agent AH</h2>
          <p style="opacity:.7;">Descreva o problema no ativo de mineração (peneira, transportador
          de correia, caminhão fora de estrada ou britador) e eu percorro o grafo de conhecimento
          para encontrar a causa provável e a ação corretiva.</p>
          <p style="opacity:.55;font-size:13px;">Ex.: <i>"O eixo excêntrico do britador está com
          vibração excessiva — qual a causa e a ação corretiva?"</i></p>
          <p style="opacity:.55;font-size:13px;">Se a pergunta não tiver detalhe suficiente
          (ex.: <i>"o motor do britador está com problema"</i>), eu pergunto qual sintoma você
          está observando antes de responder.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Renderiza o histórico da conversa ativa, incluindo o subgrafo de cada resposta.
for i, msg in enumerate(conversa["mensagens"]):
    with st.chat_message("user"):
        st.write(msg["pergunta"])
    with st.chat_message("assistant"):
        st.write(msg["resposta"])
        if msg["caminhos"]:
            titulo_expander = (
                "🔎 Opções encontradas na base — escolha uma para eu continuar"
                if msg.get("esclarecimento")
                else "🔎 Pontos conectados na base de conhecimento"
            )
            with st.expander(titulo_expander, expanded=False):
                sub = subgrafo_dos_caminhos(grafo, msg["caminhos"])
                components.html(
                    gerar_html_grafo(sub, msg["caminhos"]),
                    height=420,
                    scrolling=False,
                )

pergunta = st.chat_input("Descreva o problema do ativo...")

if pergunta:
    with st.chat_message("user"):
        st.write(pergunta)

    # Se a última resposta do agente foi um pedido de esclarecimento, a
    # pergunta original entra como contexto para o LLM combinar as duas
    # informações (ex: "motor do britador" + "sobrecorrente").
    ultima_msg = conversa["mensagens"][-1] if conversa["mensagens"] else None
    contexto_anterior = ultima_msg["pergunta"] if ultima_msg and ultima_msg.get("esclarecimento") else None

    with st.chat_message("assistant"):
        with st.spinner("Identificando entidade de partida..."):
            resultado = consultar(pergunta, grafo, contexto_anterior=contexto_anterior)

        # --- Animação de rastreio: mostra o agente "andando" pelos nós do
        # grafo que a consulta conectou, enquanto a resposta é preparada.
        # Como components.html() só EMBUTE o iframe e retorna na hora (a
        # animação em JS roda no navegador, independente do Python), ela toca
        # de verdade em paralelo com a chamada ao LLM logo abaixo.
        animacao_placeholder = st.empty()
        if resultado["caminhos"]:
            mensagem_final_animacao = (
                "Opções encontradas — escolha uma para eu continuar."
                if resultado["precisa_esclarecimento"]
                else "Caminho encontrado — gerando resposta..."
            )
            html_rastreio, n_passos = gerar_html_rastreio(
                grafo,
                resultado["caminhos"],
                resultado["entidade_partida"],
                mensagem_final=mensagem_final_animacao,
                duracao_passo_ms=int(DURACAO_PASSO_ANIMACAO_S * 1000),
            )
            with animacao_placeholder.container():
                components.html(html_rastreio, height=450, scrolling=False)

        if resultado["precisa_esclarecimento"]:
            # Não chamamos o LLM de resposta: ainda não há evidência suficiente
            # (caminho completo) para responder sem adivinhar. Como não há uma
            # chamada de rede aqui para "preencher" o tempo da animação,
            # aguardamos manualmente para o usuário ver o rastreio completo.
            duracao_estimada = min(n_passos * DURACAO_PASSO_ANIMACAO_S + 0.4, 4.0)
            time.sleep(duracao_estimada)
            resposta = resultado["pergunta_esclarecimento"]
        else:
            # A chamada ao LLM (rede) demora o suficiente para a animação
            # tocar por completo em paralelo, no navegador.
            resposta = gerar_resposta(pergunta, resultado)

        animacao_placeholder.empty()
        st.write(resposta)

        # Mostra os pontos conectados (resposta) ou as opções candidatas
        # (esclarecimento) — nunca o grafo inteiro, que fica no Sobre.
        if resultado["caminhos"]:
            titulo_expander = (
                "🔎 Opções encontradas na base — escolha uma para eu continuar"
                if resultado["precisa_esclarecimento"]
                else "🔎 Pontos conectados na base de conhecimento"
            )
            with st.expander(titulo_expander, expanded=False):
                sub = subgrafo_dos_caminhos(grafo, resultado["caminhos"])
                components.html(
                    gerar_html_grafo(sub, resultado["caminhos"]),
                    height=420,
                    scrolling=False,
                )

    conversa["mensagens"].append(
        {
            "pergunta": pergunta,
            "resposta": resposta,
            "caminhos": resultado["caminhos"],
            "esclarecimento": resultado["precisa_esclarecimento"],
        }
    )
    # A primeira pergunta vira o título da conversa na lateral (como nas LLMs).
    if conversa["titulo"] == "Nova conversa":
        conversa["titulo"] = pergunta[:45]
    st.rerun()
