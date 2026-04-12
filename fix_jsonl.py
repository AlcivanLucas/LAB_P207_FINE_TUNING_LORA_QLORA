"""
Corrige caracteres de escape inválidos nos arquivos JSONL do dataset.
Execução: python fix_jsonl.py
"""
import json

# Caracteres válidos após \ em JSON
VALID_AFTER_BACKSLASH = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}


def fix_line(line: str) -> str:
    """
    Percorre a string caractere a caractere e duplica qualquer barra
    invertida que não seja parte de uma sequência de escape JSON válida.
    """
    result = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '\\':
            next_ch = line[i + 1] if i + 1 < len(line) else ''
            if next_ch in VALID_AFTER_BACKSLASH:
                result.append(ch)        # escape válido, mantém como está
            else:
                result.append('\\\\')    # duplica a barra inválida
        else:
            result.append(ch)
        i += 1
    return ''.join(result)


def process_file(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    fixed_lines = []
    n_fixed = 0

    for i, line in enumerate(raw_lines, 1):
        line = line.rstrip('\n\r')
        if not line.strip():
            continue
        try:
            json.loads(line)
            fixed_lines.append(line)
        except json.JSONDecodeError:
            fixed = fix_line(line)
            try:
                json.loads(fixed)
                fixed_lines.append(fixed)
                n_fixed += 1
                print(f'  [OK] Linha {i} corrigida.')
            except json.JSONDecodeError as e2:
                fixed_lines.append(line)
                print(f'  [ERRO] Linha {i} não pôde ser corrigida: {e2}')

    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for ln in fixed_lines:
            f.write(ln + '\n')

    print(f'  Total: {n_fixed} linha(s) corrigida(s) — {len(fixed_lines)} linhas salvas.\n')


if __name__ == '__main__':
    for arquivo in ['dataset/dataset_train.jsonl', 'dataset/dataset_test.jsonl']:
        print(f'Processando {arquivo}...')
        process_file(arquivo)
    print('Concluído!')
