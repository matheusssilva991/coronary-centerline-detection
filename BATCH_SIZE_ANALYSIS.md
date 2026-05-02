# Análise de Tamanho de Lotes para 700 Imagens

## 🧮 Como Funcionam os Números Ímpares

O código calcula automaticamente:

```python
num_batches = (len(ids) + batch_size - 1) // batch_size  # Arredonda para cima
end_idx = min((batch_num + 1) * batch_size, len(ids))     # Limita ao máximo
```

**Exemplo com 700 imagens e batch_size=50:**

- Lote 1: índices 0-49 = 50 imagens
- Lote 2: índices 50-99 = 50 imagens
- ...
- Lote 14: índices 650-699 = 50 imagens ✓ (700 / 50 = 14 lotes exatos!)

**Exemplo com 700 imagens e batch_size=87:**

- Lote 1-8: 87 imagens cada = 696 imagens
- Lote 9: índices 696-699 = **4 imagens** (o último lote fica com o resto!)

## 📊 Melhores Opções para 700 Imagens

### Divisões Perfeitas (sem resto - RECOMENDADO)

| Batch Size | Num Lotes | Divisão | Status |
|-----------|-----------|---------|--------|
| **50**    | 14        | 700÷50  | ✅ Perfeito - Memória baixa |
| **70**    | 10        | 700÷70  | ✅ Perfeito - Balanço bom |
| **100**   | 7         | 700÷100 | ✅ Perfeito - Poucas paradas |
| **140**   | 5         | 700÷140 | ✅ Perfeito - Muito rápido |
| **175**   | 4         | 700÷175 | ✅ Perfeito - Super rápido |
| **350**   | 2         | 700÷350 | ✅ Perfeito - Só 2 lotes |

### Divisões com Resto (evitar)

| Batch Size | Num Lotes | Divisão | Problema |
|-----------|-----------|---------|----------|
| 75        | 10        | 9×75 + 25 | ⚠️ Último lote desigual |
| 80        | 9         | 8×80 + 60 | ⚠️ Último lote maior |
| 90        | 8         | 7×90 + 70 | ⚠️ Distribuição irregular |
| 87        | 9         | 8×87 + 4  | ⚠️ Último lote muito pequeno |

## 🎯 Recomendações

### 1. **Para Controle de Memória (SEGURO)**

```bash
python src/segmentation_pipeline.py --split test --batch-size 50
```

- ✅ 14 lotes pequenos e manejáveis
- ✅ Divisão perfeita
- ✅ Menos risco de crash por memória
- ⏱️ Mais tempo total (14 salvamentos)

### 2. **Balanço Ideal (RECOMENDADO)**

```bash
python src/segmentation_pipeline.py --split test --batch-size 70
```

- ✅ 10 lotes - número redondo
- ✅ Divisão perfeita (700÷70)
- ✅ Bom uso de memória
- ✅ Tempo aceitável
- ⭐ **MELHOR ESCOLHA**

### 3. **Processamento Rápido**

```bash
python src/segmentation_pipeline.py --split test --batch-size 100
```

- ✅ Apenas 7 lotes
- ✅ Divisão perfeita
- ⚠️ Requer mais memória

### 4. **Ultra Rápido**

```bash
python src/segmentation_pipeline.py --split test --batch-size 175
```

- ✅ Apenas 4 lotes
- ✅ Divisão perfeita
- ⚠️ Alto uso de memória

### 5. **Tudo de Uma Vez**

```bash
python src/segmentation_pipeline.py --split test
```

- ✅ Mais rápido
- ❌ Alto risco de crash por memória
- ❌ Sem checkpoint para retomada

## 📈 Comparação de Tempo vs Memória

```
Batch Size  | Num Lotes | Memória | Tempo Total | Retomada
------------|-----------|---------|-------------|----------
50          | 14        | Baixa   | Longo       | Fácil
70          | 10        | Média   | Médio       | Fácil ⭐
100         | 7         | Alta    | Rápido      | Fácil
175         | 4         | Muito   | Muito Rápido| Fácil
350         | 2         | Máxima  | Super Rápido| Fácil
700         | 1         | Crítico | Rápido      | Impossível ❌
```

## ⚙️ Exemplos Completos

### Execução com Retomada

```bash
# Primeira execução - processa 10 lotes de 70 imagens
python src/segmentation_pipeline.py --split test --batch-size 70 --cache

# Se falhar no lote 7:
python src/segmentation_pipeline.py --split test --batch-size 70 --resume-batch 7 --cache
# Carrega lotes 1-6, processa a partir do lote 7
```

### Com Múltiplas Resoluções

```bash
# Teste em alta resolução, lotes de 50 (mais memória necessária)
python src/segmentation_pipeline.py --split test --resolution high --batch-size 50

# Validação em resolução média, lotes de 100
python src/segmentation_pipeline.py --split val --resolution mid --batch-size 100
```

## 🔍 O que Acontece com Números Ímpares?

Com **batch-size=87** (número que não divide 700):

```
Total: 700 imagens
Batch Size: 87

Cálculo:
num_batches = (700 + 87 - 1) // 87 = 786 // 87 = 9 lotes

Distribuição:
Lote 1: imagens   1-87   = 87 imagens
Lote 2: imagens  88-174  = 87 imagens
Lote 3: imagens 175-261  = 87 imagens
Lote 4: imagens 262-348  = 87 imagens
Lote 5: imagens 349-435  = 87 imagens
Lote 6: imagens 436-522  = 87 imagens
Lote 7: imagens 523-609  = 87 imagens
Lote 8: imagens 610-696  = 87 imagens
Lote 9: imagens 697-700  = 4 imagens ⚠️ (RESTO)
```

**O último lote fica com o resto!** No exemplo acima, o lote 9 teria apenas 4 imagens.

## ✅ Conclusão

Para 700 imagens de teste:

1. **Use `--batch-size 70`** - Divisão perfeita, 10 lotes, balanço ideal
2. Alternativas boas: 50, 100 ou 175 (todas divisões perfeitas)
3. Evite números aleatórios (criarão lotes desiguais)
4. O último lote será ajustado automaticamente se necessário

## 📋 Comandos Prontos

```bash
# RECOMENDADO:
python src/segmentation_pipeline.py --split test --batch-size 70 --cache

# SEGURO (mais lotes pequenos):
python src/segmentation_pipeline.py --split test --batch-size 50 --cache

# RÁPIDO (menos lotes):
python src/segmentation_pipeline.py --split test --batch-size 100 --cache

# Em caso de falha (retomar):
python src/segmentation_pipeline.py --split test --batch-size 70 --resume-batch 7 --cache
```
