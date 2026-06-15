#!/usr/bin/env python3
"""Script para testar se scale_config_to_resolution está funcionando corretamente."""

import json
from pathlib import Path
from src.utils.config_utils import load_config_json, scale_config_to_resolution

# Carregar config padrão
config = load_config_json(Path("config/pipeline_config.json"))

print("=" * 70)
print("TESTE: scale_config_to_resolution")
print("=" * 70)

# Teste 1: MID RESOLUTION (factor_xy = 2)
print("\n[TEST 1] MID RESOLUTION (downscale_factors = [2,2,1])")
print("-" * 70)
config_mid = json.loads(json.dumps(config))  # Deep copy
config_mid["DOWNSCALE_FACTORS"] = [2, 2, 1]
scaled_mid = scale_config_to_resolution(config_mid, reference_downscale_xy=2)
print(f"Factor XY: {scaled_mid['DOWNSCALE_FACTORS'][0]}")
print(f"  radii_start_px: {scaled_mid['CIRCLE_DETECTION']['radii_start_px']} (esperado: 18)")
print(f"  radii_end_px: {scaled_mid['CIRCLE_DETECTION']['radii_end_px']} (esperado: 31)")
print(f"  num_iter: {scaled_mid['LEVEL_SET']['num_iter']} (esperado: 31)")
print(f"  threshold_divisor: {scaled_mid['REGION_GROWING']['threshold_divisor']} (esperado: 7)")

# Teste 2: HIGH RESOLUTION (factor_xy = 1)
print("\n[TEST 2] HIGH RESOLUTION (downscale_factors = [1,1,1])")
print("-" * 70)
config_high = json.loads(json.dumps(config))  # Deep copy
config_high["DOWNSCALE_FACTORS"] = [1, 1, 1]
scaled_high = scale_config_to_resolution(config_high, reference_downscale_xy=2)
print(f"Factor XY: {scaled_high['DOWNSCALE_FACTORS'][0]}")
print(f"  radii_start_px: {scaled_high['CIRCLE_DETECTION']['radii_start_px']} (esperado: 36)")
print(f"  radii_end_px: {scaled_high['CIRCLE_DETECTION']['radii_end_px']} (esperado: 62)")
print(f"  num_iter: {scaled_high['LEVEL_SET']['num_iter']} (esperado: 25 com ajuste especial)")
print(f"  threshold_divisor: {scaled_high['REGION_GROWING']['threshold_divisor']} (esperado: 12 com ajuste especial)")
print(f"  min_vesselness_fraction: {scaled_high['REGION_GROWING']['min_vesselness_fraction']} (esperado: 0.05 com ajuste especial)")

print("\n" + "=" * 70)
print("RESULTADO:")
print("=" * 70)

# Verificar se os testes passam
mid_pass = (
    scaled_mid['CIRCLE_DETECTION']['radii_start_px'] == 18 and
    scaled_mid['CIRCLE_DETECTION']['radii_end_px'] == 31 and
    scaled_mid['LEVEL_SET']['num_iter'] == 31 and
    scaled_mid['REGION_GROWING']['threshold_divisor'] == 7
)

high_pass = (
    scaled_high['CIRCLE_DETECTION']['radii_start_px'] == 36 and
    scaled_high['CIRCLE_DETECTION']['radii_end_px'] == 62 and
    scaled_high['LEVEL_SET']['num_iter'] == 25 and
    scaled_high['REGION_GROWING']['threshold_divisor'] == 12 and
    scaled_high['REGION_GROWING']['min_vesselness_fraction'] == 0.05
)

print(f"✓ MID RES: {'PASS' if mid_pass else 'FAIL'}")
print(f"✓ HIGH RES: {'PASS' if high_pass else 'FAIL'}")

if not high_pass:
    print("\n⚠️  HIGH RES scaling está com problemas!")
