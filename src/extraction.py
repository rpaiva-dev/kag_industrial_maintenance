"""Extração de entidades e relações (triplas) a partir dos documentos brutos.

Diferença fundamental para um RAG comum: em vez de fatiar o texto em chunks e
indexar por similaridade, aqui o LLM LÊ cada documento e o converte em
conhecimento ESTRUTURADO — triplas (origem, relação, destino) com tipos de
entidade. É essa estrutura que depois permite raciocínio multi-hop: seguir a
cadeia equipamento -> componente -> sintoma -> causa -> ação, algo que a busca
por similaridade não faz, porque a cadeia raramente está inteira em um único
trecho de texto.
"""

import json
from pathlib import Path

from src.llm_client import chamar_llm

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_RAW = RAIZ / "data" / "raw" / "manuais_manutencao.json"
ARQUIVO_TRIPLAS = RAIZ / "data" / "processed" / "triplas.json"

# Vocabulário fechado de relações e tipos. Um vocabulário aberto deixaria o
# LLM inventar variações ("possui_componente", "tem-componente"...) e o grafo
# ficaria fragmentado — nós que deveriam se conectar não se conectariam.
RELACOES_VALIDAS = {"tem_componente", "apresenta_sintoma", "indica_causa", "resolve_com"}
TIPOS_VALIDOS = {"equipamento", "componente", "sintoma", "causa", "acao"}

# Coerência semântica: cada relação só faz sentido entre certos tipos.
# Validar isso barra alucinações estruturais (ex: sintoma -tem_componente-> causa).
DOMINIO_RELACAO = {
    "tem_componente": ("equipamento", "componente"),
    "apresenta_sintoma": ("componente", "sintoma"),
    "indica_causa": ("sintoma", "causa"),
    "resolve_com": ("causa", "acao"),
}

# Schema para structured outputs: a API garante JSON válido nesse formato,
# eliminando a classe de erro "o LLM devolveu JSON quebrado".
SCHEMA_TRIPLAS = {
    "type": "object",
    "properties": {
        "triplas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "origem": {"type": "string"},
                    "tipo_origem": {"type": "string", "enum": sorted(TIPOS_VALIDOS)},
                    "relacao": {"type": "string", "enum": sorted(RELACOES_VALIDAS)},
                    "destino": {"type": "string"},
                    "tipo_destino": {"type": "string", "enum": sorted(TIPOS_VALIDOS)},
                },
                "required": ["origem", "tipo_origem", "relacao", "destino", "tipo_destino"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["triplas"],
    "additionalProperties": False,
}

PROMPT_SISTEMA = """Você é um extrator de conhecimento de manuais de manutenção industrial.

Sua tarefa: ler o documento e extrair TODAS as triplas (origem, relação, destino) presentes no texto, usando SOMENTE estas relações:

- tem_componente: equipamento -> componente
- apresenta_sintoma: componente -> sintoma
- indica_causa: sintoma -> causa
- resolve_com: causa -> acao

Regras:
1. Extraia apenas o que está EXPLÍCITO no texto. Não invente relações.
2. Use nomes em minúsculas para as entidades. Para sintomas, causas e ações, PRESERVE qualquer referência ao componente/equipamento presente no texto (ex: "ruído metálico no excêntrico da peneira", NUNCA encurte para apenas "ruído metálico"). Isso é crítico: um nome genérico e curto demais faz com que sintomas de EQUIPAMENTOS DIFERENTES virem, por engano, o MESMO nó do grafo — o que faria o sistema responder com dados do equipamento errado. Para componentes e equipamentos, use exatamente o nome (já qualificado) como aparece no texto.
3. Um sintoma pode indicar mais de uma causa e uma causa pode ter mais de uma ação — extraia todas.
4. Respeite a direção da relação (origem -> destino) conforme o esquema acima."""


def extrair_triplas_documento(texto: str, fonte: str) -> list[dict]:
    """Chama o LLM para um documento e devolve as triplas validadas."""
    resposta = chamar_llm(
        system=PROMPT_SISTEMA,
        user=f"Documento (fonte: {fonte}):\n\n{texto}",
        json_schema=SCHEMA_TRIPLAS,
    )
    dados = json.loads(resposta)

    triplas_validas = []
    for t in dados["triplas"]:
        # Mesmo com o schema garantindo a forma, validamos a SEMÂNTICA:
        # a relação precisa conectar os tipos certos.
        esperado = DOMINIO_RELACAO.get(t["relacao"])
        if esperado != (t["tipo_origem"], t["tipo_destino"]):
            print(f"  [descartada] tripla incoerente: {t}")
            continue
        t["origem"] = t["origem"].strip().lower()
        t["destino"] = t["destino"].strip().lower()
        t["fonte"] = fonte  # rastreabilidade: de qual documento a relação veio
        triplas_validas.append(t)
    return triplas_validas


def executar_extracao() -> list[dict]:
    """Pipeline completo: lê data/raw, extrai de cada documento, salva o JSON."""
    dados = json.loads(ARQUIVO_RAW.read_text(encoding="utf-8"))
    todas: list[dict] = []
    for doc in dados["documentos"]:
        print(f"Extraindo triplas de: {doc['fonte']}")
        triplas = extrair_triplas_documento(doc["texto"], doc["fonte"])
        print(f"  {len(triplas)} triplas extraídas")
        todas.extend(triplas)

    ARQUIVO_TRIPLAS.parent.mkdir(parents=True, exist_ok=True)
    ARQUIVO_TRIPLAS.write_text(
        json.dumps({"triplas": todas}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nTotal: {len(todas)} triplas salvas em {ARQUIVO_TRIPLAS}")
    return todas


if __name__ == "__main__":
    executar_extracao()
