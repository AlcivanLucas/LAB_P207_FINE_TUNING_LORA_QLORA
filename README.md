# Laboratório 07 — Especialização de LLMs com LoRA e QLoRA

> **Disciplina:** Inteligência Artificial Aplicada  
> **Instituição:** Instituto iCEV  
> **Aluno:** Alcivan Lucas  
> **Orientador:** Prof. Dimmy  
> **Entrega:** versão `v1.0`

---

> **Nota de Integridade Acadêmica:**  
> *"Partes geradas/complementadas com IA, revisadas por Alcivan Lucas"*

> **Uso de IA:**  
> O **Google Gemini** foi utilizado como apoio na estruturação e documentação deste projeto — incluindo organização do README, descrições técnicas e formatação da documentação. Todo o conteúdo gerado foi revisado criticamente e validado pelo aluno antes da submissão.

---

## Objetivo

Este laboratório implementa um **pipeline completo de fine-tuning** de um Large Language Model (LLM) para o domínio de **Suporte Técnico de TI**, utilizando duas técnicas fundamentais para viabilizar o treinamento em hardware com memória limitada:

| Técnica | O que faz | Biblioteca |
|---------|-----------|------------|
| **LoRA** (Low-Rank Adaptation) | Treina apenas um subconjunto mínimo de parâmetros | `peft` |
| **QLoRA** (Quantized LoRA) | Carrega o modelo em 4-bits, reduzindo o consumo de VRAM | `bitsandbytes` |

> **Modelo utilizado:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — mesma arquitetura do Llama 2, sem necessidade de licença, compatível com a GPU T4 gratuita do Google Colab.

---

## Estrutura do Projeto

```
LAB_P207_FINE_TUNING_LORA_QLORA/
│
├── dataset/
│   ├── generate_dataset.py   # Script para gerar dataset via API OpenAI
│   ├── dataset_train.jsonl   # 85 pares instrução/resposta (90% treino)
│   └── dataset_test.jsonl    #  8 pares instrução/resposta (10% teste)
│
├── finetune.py               # Pipeline principal de treinamento (QLoRA)
├── inference.py              # Script de inferência com o adaptador LoRA treinado
├── requirements.txt          # Dependências do projeto
├── .gitignore                # Arquivos a ignorar no controle de versão
└── README.md                 # Esta documentação
```

---

## Como Executar

Existem duas formas de executar este projeto: via **Google Colab** (recomendado, gratuito) ou **localmente** (requer GPU NVIDIA).

---

### Opção 1 — Google Colab (Recomendado)

> Não requer instalação local nem GPU própria. Utiliza a GPU T4 gratuita do Colab.

**Notebook pronto para execução:**  
[Abrir no Google Colab](https://colab.research.google.com/drive/1OoxoLCTVl7_aAL64UYUSgDHHi0B1WUOV?usp=sharing)

**Passo a passo:**

1. Acesse o link acima e clique em **"Abrir no Colab"**
2. Ative a GPU: `Ambiente de execução → Alterar tipo de ambiente de execução → T4 GPU`
3. Execute a célula de instalação de dependências:
   ```python
   !pip install -q transformers datasets accelerate peft trl bitsandbytes python-dotenv
   ```
4. Faça o upload dos arquivos do projeto pelo painel de arquivos (ícone de pasta):
   - `finetune.py`
   - `dataset/dataset_train.jsonl` → dentro da pasta `dataset/`
   - `dataset/dataset_test.jsonl` → dentro da pasta `dataset/`
5. Execute o fine-tuning:
   ```python
   !python finetune.py
   ```
6. O adaptador LoRA treinado será salvo em `./lora_adapter/`

**Resultado obtido no Colab (GPU T4, ~8 minutos):**

| Época | Loss (treino) | Loss (avaliação) | Acurácia (tokens) |
|-------|--------------|------------------|-------------------|
| 1 | 1.489 | 1.441 | 69.5% |
| 2 | 1.294 | 1.410 | 70.3% |
| 3 | 1.221 | 1.417 | 70.3% |

---

### Opção 2 — Execução Local

> Requer GPU NVIDIA com pelo menos 16GB de VRAM e CUDA instalado.

**Pré-requisitos de hardware:**
- GPU: NVIDIA RTX 3090 / 4090 / A100 (mínimo 16GB VRAM)
- RAM: 32GB recomendado
- Armazenamento: 20GB livres

**Pré-requisitos de software:**
- Python 3.10+
- CUDA 11.8+

**Instalação das dependências:**

```bash
pip install -r requirements.txt
```

**Passo 1 — Gerar o Dataset Sintético** *(opcional — arquivos já incluídos no repositório)*

Crie um arquivo `.env` com sua chave da OpenAI:

```
OPENAI_API_KEY=sk-...
```

Execute o script de geração:

```bash
python dataset/generate_dataset.py
```

O script gera 80 pares de instrução/resposta cobrindo 8 categorias de Suporte Técnico de TI, divide em 90% treino / 10% teste e salva em `.jsonl`.

**Passo 2 — Executar o Fine-Tuning:**

```bash
python finetune.py
```

O adaptador LoRA treinado será salvo em `./lora_adapter/`.

---

## Explicação Técnica dos Passos

### Passo 1: Engenharia de Dados Sintéticos

O dataset foi gerado com a API da OpenAI (GPT-4o-mini) cobrindo 8 categorias de Suporte Técnico de TI:

- Diagnóstico Windows
- Comandos Linux
- Redes (TCP/IP, DNS, DHCP, VLANs)
- Segurança da Informação
- Hardware e RAID
- Virtualização (Docker, Kubernetes, VMware)
- Banco de Dados (MySQL, SQL)
- Cloud e Automação (AWS, Terraform, Ansible)

**Formato dos dados (Alpaca Template):**

```
### Instrução:
Como verificar o uso de memória no Linux?

### Resposta:
Para verificar uso de memória RAM no Linux: 1) 'free -h' — exibe memória...
```

**Divisão:** 85 amostras para treino (90%) | 8 amostras para teste (10%)

---

### Passo 2: Configuração da Quantização (QLoRA)

O treinamento tradicional (*Full Fine-Tuning*) do Llama 2 7B exigiria **~112GB de VRAM**. Com QLoRA, reduzimos para **~4-6GB** com o TinyLlama 1.1B.

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # Carrega pesos em 4 bits
    bnb_4bit_quant_type="nf4",             # NormalFloat 4-bit (ideal para LLMs)
    bnb_4bit_compute_dtype=torch.bfloat16, # BFloat16 — dtype nativo do TinyLlama
    bnb_4bit_use_double_quant=False,
)
```

**Por que `nf4`?** O NormalFloat 4-bit distribui os 16 valores possíveis de forma ótima para a distribuição normal dos pesos de LLMs, minimizando o erro de quantização.

---

### Passo 3: Arquitetura LoRA

Em vez de atualizar a matriz de pesos original **W** (d×d), o LoRA injeta duas matrizes menores **A** (d×r) e **B** (r×d), onde o **rank r << d**. A saída modificada é:

```
h = Wx + (B·A)x · (alpha/r)
```

```python
lora_config = LoraConfig(
    r=64,                          # Rank: dimensão das matrizes menores
    lora_alpha=16,                 # Fator de escala (alpha/r = 0.25)
    lora_dropout=0.1,              # Dropout para evitar overfitting
    task_type=TaskType.CAUSAL_LM,  # Tarefa: modelagem de linguagem causal
)
```

**Impacto prático:**  
Sem LoRA → ~1,1 bilhão de parâmetros treináveis  
Com LoRA (r=64) → ~50 milhões de parâmetros treináveis (**4,4% do total**)

---

### Passo 4: Pipeline de Treinamento

```python
training_args = SFTConfig(
    optim="paged_adamw_32bit",   # AdamW paginado: picos de memória vão para a RAM
    lr_scheduler_type="cosine",  # Decaimento cossenoidal da taxa de aprendizado
    warmup_ratio=0.03,           # Aquece a LR nos primeiros 3% do treino
    bf16=True,                   # BFloat16 — dtype nativo do TinyLlama
    gradient_checkpointing=True, # Economiza VRAM recomputando ativações
)
```

**Por que `paged_adamw_32bit`?**  
O AdamW paginado move os estados do otimizador (m e v) para a RAM do sistema quando a GPU está sob pressão, evitando erros de `Out of Memory`.

**Por que `cosine` scheduler?**  
O decaimento cossenoidal garante que a taxa de aprendizado caia suavemente até zero, evitando que o modelo "desaprenda" nas últimas épocas.

```
LR
│ ▓▓▓
│   ▓▓▓
│      ▓▓▓▓
│          ▓▓▓▓▓▓▓▓▓▓
└────────────────────── Épocas
  ^3%^
  warmup
```

---

## Dependências Principais

| Biblioteca | Versão | Função |
|------------|--------|--------|
| `torch` | ≥2.0 | Framework de deep learning |
| `transformers` | ≥4.35 | Modelos e tokenizers Hugging Face |
| `peft` | ≥0.6 | LoRA e PEFT |
| `trl` | ≥0.7 | SFTTrainer / SFTConfig |
| `bitsandbytes` | ≥0.41 | Quantização 4-bit (QLoRA) |
| `datasets` | ≥2.14 | Carregamento dos datasets |
| `openai` | ≥1.0 | Geração do dataset sintético |

---

## Referências

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al. (2023)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. (2021)
- [Documentação PEFT — Hugging Face](https://huggingface.co/docs/peft)
- [Documentação TRL — SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [BitsAndBytes — bitsandbytes-foundation](https://github.com/TimDettmers/bitsandbytes)
- [TinyLlama — Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
