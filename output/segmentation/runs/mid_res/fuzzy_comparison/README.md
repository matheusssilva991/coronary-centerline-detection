# Comparação dos métodos fuzzy

Esta pasta reúne as execuções usadas para comparar os métodos de threshold e
segmentação arterial em resolução média.

## Estrutura

```text
fuzzy_comparison/
  train/
    <variant>/<timestamp>/
  val/
    <variant>/<timestamp>/
  test/
    <variant>/<timestamp>/
```

Cada execução mantém a estrutura padrão com `config/`, `numeric/` e, quando
disponível, `logs/`.

## Variantes

- `normal_rg`: threshold normal com region growing.
- `normal_fc`: threshold normal com fuzzy connectedness.
- `th_fuzzy_rg`: threshold fuzzy com region growing.
- `th_fuzzy_fc`: threshold fuzzy com fuzzy connectedness.

O split aparece antes da variante para deixar explícito quais resultados podem
ser usados para ajuste (`train` e `val`) e quais devem permanecer reservados
para avaliação final (`test`).
