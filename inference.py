"""
===========================================================================
Inferência com o Modelo Fine-Tunado (LoRA Adapter)
===========================================================================

Script para testar o modelo Llama 2 7B após o fine-tuning com QLoRA.
Carrega o modelo base com quantização 4-bit e aplica o adaptador LoRA
treinado, permitindo testar respostas no domínio de Suporte Técnico de TI.

Uso:
    python inference.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
MODEL_ID         = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH     = "./lora_adapter"
MAX_NEW_TOKENS   = 512
TEMPERATURE      = 0.7
TOP_P            = 0.9

# Reutiliza a mesma configuração de quantização do treinamento
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


def carregar_modelo():
    """Carrega o modelo base + adaptador LoRA."""
    print("Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    print("Carregando modelo base com quantização 4-bit...")
    modelo_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Aplicando adaptador LoRA de '{ADAPTER_PATH}'...")
    modelo = PeftModel.from_pretrained(modelo_base, ADAPTER_PATH)
    modelo.eval()

    print("Modelo pronto!\n")
    return modelo, tokenizer


def gerar_resposta(modelo, tokenizer, instrucao: str) -> str:
    """Gera uma resposta para uma instrução de suporte técnico."""
    prompt = (
        "### Instrução:\n"
        f"{instrucao}\n\n"
        "### Resposta:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(modelo.device)

    with torch.no_grad():
        outputs = modelo.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Remove o prompt da saída gerada
    resposta_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    resposta = tokenizer.decode(resposta_tokens, skip_special_tokens=True)
    return resposta.strip()


def main():
    modelo, tokenizer = carregar_modelo()

    # Exemplos de teste no domínio de Suporte Técnico de TI
    perguntas_teste = [
        "Como verificar o uso de CPU em tempo real no Linux?",
        "Qual o procedimento para liberar espaço em disco no Windows sem perder arquivos?",
        "Como configurar um IP estático em uma máquina com Ubuntu Server?",
    ]

    print("=" * 70)
    print("  Testando o modelo fine-tunado — Suporte Técnico de TI")
    print("=" * 70)

    for i, pergunta in enumerate(perguntas_teste, 1):
        print(f"\n[Pergunta {i}]: {pergunta}")
        print("-" * 50)
        resposta = gerar_resposta(modelo, tokenizer, pergunta)
        print(f"[Resposta]: {resposta}")
        print()


if __name__ == "__main__":
    main()
