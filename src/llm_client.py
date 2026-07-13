"""Cliente centralizado para o LLM (OpenAI).

Por que centralizar? Todos os módulos (extração, identificação de entidade,
geração de resposta) precisam chamar o LLM. Concentrar a criação do cliente,
o carregamento da chave e o modelo padrão em um único lugar evita duplicação
e facilita trocar de modelo ou provedor depois — os demais módulos só
conhecem a função chamar_llm().
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Modelo padrão do projeto. Um só lugar para trocar quando necessário.
MODELO_PADRAO = "gpt-4o"

_cliente: OpenAI | None = None


def _obter_chave_api() -> str | None:
    """Resolve a chave da API: primeiro .env local, depois st.secrets (produção).

    O import do streamlit é feito de forma preguiçosa (dentro da função) porque
    os scripts offline (extraction, graph_builder) rodam fora do Streamlit e
    não devem depender dele.
    """
    load_dotenv()  # carrega .env se existir; não sobrescreve variáveis já definidas
    chave = os.getenv("OPENAI_API_KEY")
    if chave:
        return chave
    try:
        import streamlit as st

        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def obter_cliente() -> OpenAI:
    """Retorna um cliente singleton. Falha cedo, com mensagem clara, se não há chave."""
    global _cliente
    if _cliente is None:
        chave = _obter_chave_api()
        if not chave:
            raise RuntimeError(
                "OPENAI_API_KEY não encontrada. Crie um arquivo .env "
                "(veja .env.example) ou configure st.secrets."
            )
        _cliente = OpenAI(api_key=chave)
    return _cliente


def chamar_llm(
    system: str,
    user: str,
    max_tokens: int = 4096,
    json_schema: dict | None = None,
) -> str:
    """Faz uma chamada simples ao LLM e retorna o texto da resposta.

    json_schema: quando fornecido, usa structured outputs (response_format
    json_schema com strict=True) para GARANTIR que a resposta é um JSON válido
    aderente ao schema — mais robusto do que pedir JSON no prompt e torcer.
    """
    cliente = obter_cliente()

    kwargs: dict = dict(
        model=MODELO_PADRAO,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if json_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "saida", "strict": True, "schema": json_schema},
        }

    resposta = cliente.chat.completions.create(**kwargs)
    return resposta.choices[0].message.content or ""
