"""
Script para comparar performance entre GPU e CPU no Frangi filter.
Carrega uma imagem real e mostra ganho de performance ao usar GPU.
"""

import numpy as np
import time
import os
import sys
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils.frangi import get_vesselness, use_gpu
from utils import load_raw_img_and_label, run_core_preprocessing_pipeline

print("=" * 60)
print("COMPARAÇÃO GPU vs CPU - Frangi Filter")
print("=" * 60)

gpu_status = use_gpu()
print(f"GPU Disponível: {gpu_status}")
if not gpu_status:
    print("⚠ GPU não detectada ou cuCIM com problemas.")
    print("  Apenas CPU será usado.\n")
else:
    print("✓ GPU e cuCIM prontos!\n")

# Carrega imagem real
print("Carregando imagem real...")
base_path = "/data04/home/mpmaia/ImageCAS/database/1-1000"
IMG_ID = 1  # ID da imagem a testar

try:
    nii_img, nii_label = load_raw_img_and_label(
        f"{base_path}/{IMG_ID}.img.nii.gz",
        f"{base_path}/{IMG_ID}.label.nii.gz"
    )

    spacing = nii_img.header.get_zooms()
    img = np.array(nii_img.get_fdata())

    print(f"✓ Imagem carregada: ID={IMG_ID}")
    print(f"  Shape: {img.shape}")
    print(f"  Spacing: {spacing[:3]}")
    print(f"  Range: [{img.min():.1f}, {img.max():.1f}]")
    print(f"  Tamanho: {img.nbytes / 1024 / 1024:.1f} MB\n")

    # Pré-processa a imagem
    print("Pré-processando imagem...")
    downscale_factors = (2, 2, 1)
    down_image, thresh_image, lcc_image, thresh_vals = run_core_preprocessing_pipeline(
        img,
        downscale_factors=downscale_factors,
        lcc_per_slice=True,
        max_threshold_percentile=99.5
    )

    test_image = lcc_image
    print(f"✓ Imagem pré-processada: {test_image.shape}")
    print(f"  Tamanho: {test_image.nbytes / 1024 / 1024:.1f} MB\n")

except FileNotFoundError:
    print(f"⚠ Imagem não encontrada em {base_path}/{IMG_ID}.img.nii.gz")
    print("Criando imagem de teste sintética em vez disso...")
    test_image = np.random.rand(128, 128, 128).astype(np.float32) * 255
    print(f"Tamanho: {test_image.nbytes / 1024 / 1024:.1f} MB\n")

params = {
    "normalization": "robust",
    "alpha": 0.5,
    "beta": 0.5,
}

# Test CPU
print("-" * 60)
print("CPU Processing:")
print("-" * 60)
start = time.time()
result_cpu = get_vesselness(test_image, gpu=False, **params)
cpu_time = time.time() - start
print(f"⏱  Tempo: {cpu_time:.2f}s")
print(f"📊 Resultado shape: {result_cpu.shape}")
print(f"💾 Tipo: {type(result_cpu).__name__}\n")

# Test GPU (se disponível)
if use_gpu():
    print("-" * 60)
    print("GPU Processing:")
    print("-" * 60)
    start = time.time()
    result_gpu = get_vesselness(test_image, gpu=True, **params)
    gpu_time = time.time() - start
    print(f"⏱  Tempo: {gpu_time:.2f}s")
    print(f"📊 Resultado shape: {result_gpu.shape}")
    print(f"💾 Tipo: {type(result_gpu).__name__}\n")

    # Comparação
    print("=" * 60)
    print("RESULTADOS:")
    print("=" * 60)
    speedup = cpu_time / gpu_time
    improvement = (cpu_time - gpu_time) / cpu_time * 100
    print(f"🚀 Speedup: {speedup:.1f}x")
    print(f"📈 Melhoria: {improvement:.1f}%")
    print(f"\nCPU:  {cpu_time:.2f}s")
    print(f"GPU:  {gpu_time:.2f}s")
    print(f"Economia: {cpu_time - gpu_time:.2f}s\n")

    # Verifica se resultados são similares
    max_diff = np.abs(result_cpu - result_gpu).max()
    mean_diff = np.abs(result_cpu - result_gpu).mean()
    print(f"✓ Máxima diferença: {max_diff:.6f}")
    print(f"✓ Diferença média: {mean_diff:.6f}")
    if max_diff < 1e-5:
        print("✓ Resultados são idênticos!")
else:
    print("⚠ GPU não disponível.")
    print("  Possíveis causas:")
    print("  - CuPy não instalado: pip install cupy-cuda12x")
    print("  - cuCIM não instalado: pip install cucim")
    print("  - Bug no cuCIM (KeyError 'area'): versão incompatível")
    print("  - Nenhuma GPU CUDA disponível no sistema")

print("\n" + "=" * 60)
