#!/usr/bin/env python3
"""
Script para analisar melhores tamanhos de lote para um dado número de imagens.
"""

import math
from typing import List, Tuple


def analyze_batch_sizes(
    num_images: int, min_batch: int = 10, max_batch: int = None
) -> List[Tuple[int, int, bool]]:
    """
    Analisa divisões de lotes para um número de imagens.

    Args:
        num_images: Total de imagens
        min_batch: Mínimo tamanho de lote a considerar
        max_batch: Máximo tamanho de lote (padrão: num_images)

    Returns:
        Lista de (batch_size, num_lotes, é_perfeito)
    """
    if max_batch is None:
        max_batch = num_images

    results = []
    for batch_size in range(min_batch, max_batch + 1):
        num_batches = math.ceil(num_images / batch_size)
        is_perfect = (num_images % batch_size) == 0
        results.append((batch_size, num_batches, is_perfect))

    return results


def print_analysis(num_images: int):
    """Imprime análise formatada."""
    print(f"\n{'=' * 70}")
    print(f"ANÁLISE DE LOTES PARA {num_images} IMAGENS")
    print(f"{'=' * 70}\n")

    # Encontrar divisões perfeitas
    perfect_divisions = []
    for i in range(1, num_images + 1):
        if num_images % i == 0:
            perfect_divisions.append(i)

    print("📌 DIVISÕES PERFEITAS (SEM RESTO):\n")
    print(f"{'Batch Size':<12} {'Num Lotes':<12} {'Status':<20}")
    print("-" * 70)

    # Filtrar divisões interessantes (até 10 lotes no máximo)
    interesting_perfect = [d for d in perfect_divisions if num_images // d <= 20]

    for batch_size in interesting_perfect:
        num_lotes = num_images // batch_size
        if num_lotes <= 20:  # Mostrar apenas divisões razoáveis
            status = "✅ PERFEITO"
            print(f"{batch_size:<12} {num_lotes:<12} {status:<20}")

    print("\n" + "=" * 70)
    print("⚠️  DIVISÕES COM RESTO (evitar):\n")
    print(f"{'Batch Size':<12} {'Num Lotes':<12} {'Resto':<12} {'Status':<20}")
    print("-" * 70)

    # Mostrar algumas divisões com resto
    test_sizes = [40, 60, 75, 80, 90, 125, 150, 200]
    for batch_size in test_sizes:
        if batch_size <= num_images:
            num_lotes = math.ceil(num_images / batch_size)
            resto = num_images % batch_size
            if resto > 0:
                status = "⚠️  Com resto"
                print(f"{batch_size:<12} {num_lotes:<12} {resto:<12} {status:<20}")

    print("\n" + "=" * 70)
    print("🎯 RECOMENDAÇÕES:\n")

    # Recomendar 3 melhores opções
    candidates = []
    for batch_size in interesting_perfect:
        num_lotes = num_images // batch_size
        if 5 <= num_lotes <= 20:  # Bom balanço
            candidates.append((batch_size, num_lotes))
        elif num_lotes < 5:
            candidates.append((batch_size, num_lotes, "rápido"))

    if candidates:
        print("1️⃣  RECOMENDADO (balanço ideal):")
        batch_size, num_lotes = candidates[0][:2]
        print(
            f"   python src/segmentation_pipeline.py --split test --batch-size {batch_size}"
        )
        print(f"   → {num_lotes} lotes de {batch_size} imagens\n")

        if len(candidates) > 1:
            print("2️⃣  ALTERNATIVA (mais seguro):")
            batch_size, num_lotes = candidates[1][:2]
            print(
                f"   python src/segmentation_pipeline.py --split test --batch-size {batch_size}"
            )
            print(f"   → {num_lotes} lotes de {batch_size} imagens\n")

        if len(candidates) > 2:
            print("3️⃣  ALTERNATIVA (mais rápido):")
            batch_size, num_lotes = candidates[2][:2]
            print(
                f"   python src/segmentation_pipeline.py --split test --batch-size {batch_size}"
            )
            print(f"   → {num_lotes} lotes de {batch_size} imagens\n")

    print("=" * 70 + "\n")


def show_last_batch_example(num_images: int, batch_size: int):
    """Mostra exemplo do que acontece no último lote."""
    print(f"\n{'=' * 70}")
    print(f"EXEMPLO PRÁTICO: {num_images} imagens com batch-size={batch_size}")
    print(f"{'=' * 70}\n")

    num_lotes = math.ceil(num_images / batch_size)
    resto = num_images % batch_size

    print(f"Total: {num_images} imagens")
    print(f"Batch Size: {batch_size}")
    print(f"Número de Lotes: {num_lotes}\n")

    print("Distribuição:\n")
    for lote in range(1, num_lotes + 1):
        start_idx = (lote - 1) * batch_size
        end_idx = min(lote * batch_size, num_images)
        tamanho_lote = end_idx - start_idx

        status = "✅" if tamanho_lote == batch_size else "⚠️ (RESTO)"
        print(
            f"  Lote {lote}: imagens {start_idx + 1:4d}-{end_idx:4d} = {tamanho_lote:3d} imagens {status}"
        )

    if resto > 0:
        print(
            f"\n⚠️  Atenção: O último lote tem apenas {resto} imagens (resto da divisão)"
        )
        print(
            f"   Total processado: {num_lotes - 1} × {batch_size} + {resto} = {num_images}"
        )
    else:
        print(
            f"\n✅ Divisão Perfeita! Todos os {num_lotes} lotes têm exatamente {batch_size} imagens"
        )

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    # Análise para 700 imagens
    print_analysis(700)

    # Exemplos práticos
    show_last_batch_example(700, 70)
    show_last_batch_example(700, 87)
    show_last_batch_example(700, 50)
