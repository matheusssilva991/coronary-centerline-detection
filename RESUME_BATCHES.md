# Referência Rápida: Retomada de Lotes

## ✅ Problema Resolvido

Agora quando você retoma um processamento em lote, **o código usa o mesmo diretório anterior**, em vez de criar um novo.

## 📋 Como Usar

### 1️⃣ Primeira Execução (cria novo diretório)

```bash
python src/segmentation_pipeline.py --split test --batch-size 70 --cache
```

**Saída:**

```
📁 Diretório de saída: output/segmentation/2026-03-14_10-30-00

📦 Processando lote 1/10 (70 imagens)
...
```

### 2️⃣ Se Falhar em um Lote

Se o processamento falhar no **lote 3**, retome com:

```bash
python src/segmentation_pipeline.py --split test --batch-size 70 --resume-batch 3 --resume-dir output/segmentation/2026-03-14_10-30-00 --cache
```

**O que acontece:**

- ✅ Carrega os lotes 1 e 2 dos CSVs anteriores
- ✅ **Processa a partir do lote 3**
- ✅ **Salva no MESMO diretório** (não cria novo)
- ✅ Após terminar, mescla todos os lotes

### 3️⃣ Verificar Resultados

```bash
# Ver arquivos salvos
ls -la output/segmentation/2026-03-14_10-30-00/

# Deve ter:
# - ostios_test_lote_1.csv (carregado)
# - ostios_test_lote_2.csv (carregado)
# - ostios_test_lote_3.csv (novo - processado)
# - ostios_test_lote_4.csv (novo - processado)
# ...
# - ostios_test_summary.csv (mesclado ao final)
```

## 🎯 Comandos Prontos

### Cenário 1: Falhou no lote 3

```bash
# Primeiro
python src/segmentation_pipeline.py --split test --batch-size 70 --cache

# Se falhar, retomar no lote 3:
python src/segmentation_pipeline.py --split test --batch-size 70 --resume-batch 3 --resume-dir output/segmentation/2026-03-14_10-30-00 --cache
```

### Cenário 2: Com 700 imagens, lotes de 70

```bash
# Primeira execução
python src/segmentation_pipeline.py --split test --batch-size 70

# Listar diretório criado
ls -d output/segmentation/*/ | tail -1

# Se falhar, pegue o diretório e faça:
python src/segmentation_pipeline.py --split test --batch-size 70 --resume-batch 7 --resume-dir output/segmentation/2026-03-14_10-30-00
```

### Cenário 3: Múltiplos splits

```bash
# Processar train, val e test em lotes
python src/segmentation_pipeline.py --split train val test --batch-size 100 --cache

# Se falhar em um, retomar o mesmo:
python src/segmentation_pipeline.py --split train --batch-size 100 --resume-batch 5 --resume-dir output/segmentation/2026-03-14_10-30-00 --cache
```

## ⚠️ Importante

1. **Batch size deve ser o mesmo** quando retoma
2. **Split deve ser o mesmo** quando retoma
3. O **diretório anterior** deve existir (com os lotes já processados)
4. Ao retomar, os lotes anteriores são **carregados dos CSVs** automaticamente

## 🔄 Fluxo Completo

```
[Primeira Execução]
python ... --batch-size 70
├─ Cria: output/segmentation/2026-03-14_10-30-00/
├─ Processa lotes 1-10
├─ Salva: lote_1.csv, lote_2.csv, ..., lote_10.csv
└─ Mescla: summary.csv

[Se Falhar no Lote 7]
python ... --resume-batch 7 --resume-dir output/segmentation/2026-03-14_10-30-00/
├─ Carrega lotes 1-6 dos CSVs
├─ Processa lotes 7-10
├─ Salva: lote_7.csv, lote_8.csv, ..., lote_10.csv
└─ Mescla: summary.csv (consolidado com todos)

[Resultado Final]
output/segmentation/2026-03-14_10-30-00/
├─ ostios_test_lote_1.csv ✅
├─ ostios_test_lote_2.csv ✅
├─ ostios_test_lote_3.csv ✅
├─ ...
├─ ostios_test_lote_10.csv ✅
└─ ostios_test_summary.csv ✅ (consolidado)
```

## 📊 Exemplo Prático com 700 Imagens

```bash
# 1. Primeira vez (10 lotes de 70 imagens cada)
python src/segmentation_pipeline.py --split test --batch-size 70 --cache

# Saída: output/segmentation/2026-03-14_10-30-00/

# 2. Se falhar no lote 7:
python src/segmentation_pipeline.py --split test --batch-size 70 \
  --resume-batch 7 \
  --resume-dir output/segmentation/2026-03-14_10-30-00 \
  --cache

# 3. Verificar resultado
ls -lh output/segmentation/2026-03-14_10-30-00/*.csv
```

## ✨ Resumo

- ✅ **Mesmo diretório**: Usa `--resume-dir`
- ✅ **Sem novo timestamp**: Não cria pasta nova
- ✅ **Lotes automáticos**: Carrega lotes anteriores do CSV
- ✅ **Merge automático**: Consolida ao final
