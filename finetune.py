"""
===========================================================================
Laboratório 07 - Especialização de LLMs com LoRA e QLoRA
Instituto iCEV - Curso de Inteligência Artificial
===========================================================================

Pipeline completo de fine-tuning do modelo Llama 2 7B com QLoRA.

Passos implementados:
  Passo 2 — Quantização 4-bit com BitsAndBytes (nf4 + float16)
  Passo 3 — Arquitetura LoRA  (r=64, alpha=16, dropout=0.1)
  Passo 4 — Treinamento com SFTTrainer (paged_adamw_32bit, cosine scheduler)

Uso:
    python finetune.py

Requisitos de hardware:
    GPU com mínimo 16GB VRAM (ex: NVIDIA A100, RTX 3090/4090)
    ou use Google Colab Pro+ / Kaggle com GPU T4/A100
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Configurações Gerais
# ---------------------------------------------------------------------------

# Modelo base (Llama 2 7B — requer aceitação de licença em huggingface.co/meta-llama)
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Caminhos dos datasets gerados no Passo 1
TRAIN_FILE = "dataset/dataset_train.jsonl"
TEST_FILE  = "dataset/dataset_test.jsonl"

# Diretório onde o adaptador LoRA será salvo
OUTPUT_DIR       = "./results"
ADAPTER_SAVE_DIR = "./lora_adapter"

# Comprimento máximo de sequência (tokens)
MAX_SEQ_LENGTH = 512

# Número de épocas de treinamento
NUM_EPOCHS = 3


# ===========================================================================
# PASSO 2: Configuração da Quantização (QLoRA)
# ===========================================================================
# O BitsAndBytesConfig carrega o modelo em 4 bits, reduzindo drasticamente
# o uso de VRAM. Usamos nf4 (NormalFloat 4-bit), que é o tipo de quantização
# mais adequado para pesos de LLMs. O compute_dtype define a precisão dos
# cálculos durante o forward/backward pass.
# ===========================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # Ativa a quantização 4-bit
    bnb_4bit_quant_type="nf4",               # NormalFloat 4-bit (melhor para LLMs)
    bnb_4bit_compute_dtype=torch.bfloat16,   # BFloat16 — dtype nativo do TinyLlama
    bnb_4bit_use_double_quant=False,         # Double quantization (economiza +0,4 bits/param)
)


# ===========================================================================
# PASSO 3: Arquitetura LoRA
# ===========================================================================
# O LoRA (Low-Rank Adaptation) congela os pesos originais do modelo e injeta
# matrizes de decomposição de baixo rank (A e B) em camadas de atenção.
# Em vez de atualizar W (d×d), atualizamos A (d×r) e B (r×d) onde r << d.
# Isso reduz drasticamente o número de parâmetros treináveis.
#
# Parâmetros obrigatórios:
#   r=64     → dimensão das matrizes menores (rank)
#   alpha=16 → fator de escala; os pesos são escalonados por (alpha/r)
#   dropout  → aplicado nas matrizes LoRA para evitar overfitting
# ===========================================================================

lora_config = LoraConfig(
    r=64,                        # Rank: dimensão das matrizes de decomposição
    lora_alpha=16,               # Alpha: fator de escala dos novos pesos
    lora_dropout=0.1,            # Dropout para regularização
    bias="none",                 # Não treinar os termos de bias
    task_type=TaskType.CAUSAL_LM,  # Tipo de tarefa: modelagem de linguagem causal
    target_modules=[             # Camadas onde o LoRA será injetado
        "q_proj",                # Projeção de query na atenção
        "v_proj",                # Projeção de value na atenção
        "k_proj",                # Projeção de key
        "o_proj",                # Projeção de output
        "gate_proj",             # FFN gate
        "up_proj",               # FFN up projection
        "down_proj",             # FFN down projection
    ],
)


# ===========================================================================
# PASSO 4: Pipeline de Treinamento e Otimização
# ===========================================================================
# Usamos o SFTTrainer (Supervised Fine-Tuning Trainer) da biblioteca trl,
# que simplifica o fine-tuning instrucional em comparação ao Trainer padrão.
#
# Configurações de otimização de memória:
#   paged_adamw_32bit → AdamW paginado: transfere picos de memória da GPU
#                       para a RAM do sistema via paginação, evitando OOM
#   cosine scheduler  → A taxa de aprendizado decai suavemente seguindo uma
#                       curva cosseno, evitando colapso no final do treino
#   warmup_ratio=0.03 → Os primeiros 3% do treino aumentam a LR gradualmente
# ===========================================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # --- Épocas e batch ---
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,

    # --- Otimizador e agendamento da taxa de aprendizado ---
    optim="paged_adamw_32bit",       # AdamW paginado (reduz picos de VRAM)
    learning_rate=2e-4,
    lr_scheduler_type="cosine",      # Decaimento cossenoidal da taxa de aprendizado
    warmup_ratio=0.03,               # 3% do treino para aquecimento gradual

    # --- Eficiência de memória ---
    bf16=True,                       # BFloat16 — dtype nativo do TinyLlama (substitui fp16)
    group_by_length=True,            # Agrupa sequências de tamanho similar (menos padding)
    gradient_checkpointing=True,     # Recomputa ativações no backward (economiza VRAM)

    # --- Avaliação e salvamento ---
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,              # Mantém apenas os 2 melhores checkpoints

    # --- Logging ---
    logging_steps=10,
    report_to="none",                # Desativa WandB por padrão; mude para "wandb" se quiser
)


def formatar_instrucao(exemplo: dict) -> str:
    """
    Formata um par (instruction, response) no template Alpaca.
    Este formato instrucional é o mais utilizado para fine-tuning
    supervisionado de modelos de linguagem.
    """
    return (
        "### Instrução:\n"
        f"{exemplo['instruction']}\n\n"
        "### Resposta:\n"
        f"{exemplo['response']}"
    )


def main():
    print("=" * 70)
    print("  Fine-Tuning com QLoRA — Suporte Técnico de TI")
    print("  Laboratório 07 — iCEV")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Carregar o Tokenizer
    # ------------------------------------------------------------------
    print("\n[1/6] Carregando o tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Necessário para evitar warnings com fp16

    # ------------------------------------------------------------------
    # 2. Carregar o Modelo Base com Quantização 4-bit (Passo 2)
    # ------------------------------------------------------------------
    print("[2/6] Carregando o modelo base com quantização 4-bit (QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",           # Distribui automaticamente entre GPUs disponíveis
        trust_remote_code=True,
    )
    model.config.use_cache = False          # Desabilita o KV cache durante treino
    model.config.pretraining_tp = 1        # Tensor parallelism = 1 para treino simples

    # Prepara o modelo quantizado para treinamento com LoRA
    model = prepare_model_for_kbit_training(model)

    # ------------------------------------------------------------------
    # 3. Configuração LoRA (Passo 3) — aplicada pelo SFTTrainer internamente
    # ------------------------------------------------------------------
    print("[3/6] Configuração LoRA definida (r=64, alpha=16, dropout=0.1) — será aplicada pelo SFTTrainer...")

    # ------------------------------------------------------------------
    # 4. Carregar os Datasets
    # ------------------------------------------------------------------
    print("[4/6] Carregando os datasets JSONL...")
    dataset_treino = load_dataset("json", data_files=TRAIN_FILE, split="train")
    dataset_teste  = load_dataset("json", data_files=TEST_FILE,  split="train")
    print(f"      Treino: {len(dataset_treino)} amostras | Teste: {len(dataset_teste)} amostras")

    # ------------------------------------------------------------------
    # 5. Configurar e Executar o SFTTrainer (Passo 4)
    # ------------------------------------------------------------------
    print("[5/6] Configurando o SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_treino,
        eval_dataset=dataset_teste,
        peft_config=lora_config,
        formatting_func=formatar_instrucao,   # Aplica o template Alpaca
        processing_class=tokenizer,
        args=training_args,
    )

    print("[5/6] Iniciando o treinamento...")
    trainer.train()

    # ------------------------------------------------------------------
    # 6. Salvar o Adaptador LoRA
    # ------------------------------------------------------------------
    print(f"\n[6/6] Salvando o adaptador LoRA em '{ADAPTER_SAVE_DIR}'...")
    trainer.model.save_pretrained(ADAPTER_SAVE_DIR)
    tokenizer.save_pretrained(ADAPTER_SAVE_DIR)

    print("\n" + "=" * 70)
    print("  Treinamento concluído com sucesso!")
    print(f"  Adaptador LoRA salvo em: {ADAPTER_SAVE_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
