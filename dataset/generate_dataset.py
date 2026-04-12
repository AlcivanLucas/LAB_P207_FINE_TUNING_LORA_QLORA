"""
===========================================================
Passo 1: Engenharia de Dados Sintéticos (Dataset Generation)
===========================================================

Script para gerar um dataset sintético de instruções no domínio
de Suporte Técnico de TI, utilizando a API da OpenAI (GPT-4).

Gera pelo menos 80 pares (instruction, response) e divide em:
  - 90% treino  → dataset_train.jsonl
  - 10%  teste  → dataset_test.jsonl

Uso:
    Crie um arquivo .env com:   OPENAI_API_KEY=sk-...
    Então execute:              python dataset/generate_dataset.py
"""

import os
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ------------------------------------------------------------
# Configuração
# ------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TOTAL_SAMPLES   = 80        # Total de pares a gerar
TRAIN_RATIO     = 0.9       # 90% para treino
RANDOM_SEED     = 42
OUTPUT_DIR      = Path(__file__).parent

# Domínio do dataset
DOMAIN = "Suporte Técnico de TI"

# Categorias de perguntas para cobrir o domínio de forma abrangente
CATEGORIES = [
    "Diagnóstico e resolução de problemas no Windows",
    "Comandos essenciais do Linux para administradores",
    "Configuração e troubleshooting de redes (TCP/IP, DNS, DHCP)",
    "Segurança da informação e boas práticas",
    "Hardware: diagnóstico de componentes e manutenção",
    "Virtualização com VMware e VirtualBox",
    "Banco de dados: SQL, backup e recuperação",
    "Cloud computing: AWS, Azure e GCP básico",
]

SYSTEM_PROMPT = """Você é um especialista em suporte técnico de TI com mais de 15 anos
de experiência. Sua tarefa é gerar pares de instrução/resposta realistas para treinar
um modelo de linguagem especializado em suporte técnico.

Cada par deve:
1. Ter uma instrução clara (pergunta ou pedido de um usuário/técnico)
2. Ter uma resposta técnica, detalhada e prática
3. Usar terminologia técnica correta em português brasileiro
4. Ser útil para técnicos de nível júnior a sênior

Retorne APENAS um JSON no formato:
{
  "instruction": "...",
  "response": "..."
}"""


def generate_pair(client: OpenAI, category: str) -> dict:
    """Gera um par instruction/response via API da OpenAI."""
    user_prompt = (
        f"Crie um par instrução/resposta sobre: {category}\n"
        "A instrução deve simular uma dúvida ou problema real de suporte técnico."
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.8,
        max_tokens=600,
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content
    pair = json.loads(content)

    # Garantir que as chaves corretas existem
    assert "instruction" in pair and "response" in pair, \
        f"Resposta inválida da API: {pair}"

    return pair


def generate_dataset(n_samples: int = TOTAL_SAMPLES) -> list[dict]:
    """Gera n_samples pares distribuídos pelas categorias."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    dataset = []

    samples_per_category = n_samples // len(CATEGORIES)
    remainder = n_samples % len(CATEGORIES)

    for i, category in enumerate(CATEGORIES):
        n = samples_per_category + (1 if i < remainder else 0)
        print(f"  Gerando {n} amostras para: '{category}'...")

        for j in range(n):
            try:
                pair = generate_pair(client, category)
                dataset.append(pair)
                print(f"    [{j + 1}/{n}] OK")
            except Exception as e:
                print(f"    [{j + 1}/{n}] ERRO: {e}")

    return dataset


def split_and_save(dataset: list[dict], train_ratio: float = TRAIN_RATIO):
    """Embaralha, divide e salva os arquivos .jsonl."""
    random.seed(RANDOM_SEED)
    random.shuffle(dataset)

    split_idx  = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    test_data  = dataset[split_idx:]

    train_path = OUTPUT_DIR / "dataset_train.jsonl"
    test_path  = OUTPUT_DIR / "dataset_test.jsonl"

    for path, data in [(train_path, train_data), (test_path, test_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  Treino : {len(train_data)} amostras → {train_path}")
    print(f"  Teste  : {len(test_data)} amostras  → {test_path}")
    print(f"  Total  : {len(dataset)} amostras")


if __name__ == "__main__":
    print("=" * 60)
    print("  Geração de Dataset Sintético - Suporte Técnico de TI")
    print("=" * 60)

    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "Variável OPENAI_API_KEY não encontrada.\n"
            "Crie um arquivo .env com: OPENAI_API_KEY=sk-..."
        )

    print(f"\nGerando {TOTAL_SAMPLES} pares de instrução/resposta...\n")
    dataset = generate_dataset(TOTAL_SAMPLES)

    print("\nSalvando os arquivos .jsonl...")
    split_and_save(dataset)

    print("\nConcluído com sucesso!")
