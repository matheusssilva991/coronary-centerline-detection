"""Argumentos de linha de comando para o pipeline de segmentação."""

from __future__ import annotations

import argparse
from pathlib import Path


PIPELINE_EPILOG = """
Exemplos de uso:
  # Processar todos os conjuntos
  python segmentation_pipeline.py

  # Processar apenas treino
  python segmentation_pipeline.py --split train

  # Processar validação e teste
  python segmentation_pipeline.py --split val test

  # Usar OpenCV para downscaling com interpolação AREA
  python segmentation_pipeline.py --downscale-method opencv

  # Usar OpenCV com interpolação LINEAR
  python segmentation_pipeline.py --downscale-method opencv --opencv-interpolation linear

  # Usar resolução alta (sem downscaling)
  python segmentation_pipeline.py --resolution high

  # Usar resolução média (downscale 2x)
  python segmentation_pipeline.py --resolution mid --split val

  # Escolher método de segmentação arterial
  python segmentation_pipeline.py --split val --artery-method fc

  # Usar a seleção bilateral com superfície fina e correção condicional da aorta
  python segmentation_pipeline.py --split val --aorta-ostia-method bilateral_thin

  # Manter/restaurar a abordagem histórica de aorta e óstios
  python segmentation_pipeline.py --split val --aorta-ostia-method standard

  # Ativar o controle adaptativo das iterações do level set da aorta
  python segmentation_pipeline.py --split train --aorta-level-set-mode adaptive

  # Comparar candidatos do region growing com a média acumulada da região
  python segmentation_pipeline.py --split train --rg-comparison-window -1

  # Usar fuzzy threshold
  python segmentation_pipeline.py --split val --threshold-method fuzzy --lower-threshold-percentile 10.5

  # Testar limiar inferior adaptativo mantendo threshold normal + RG
  python segmentation_pipeline.py --split train --threshold-method normal --artery-method rg --lower-threshold-method percentile --lower-threshold-percentile 10.75

  # Sobrescrever apenas o percentil superior do threshold
  python segmentation_pipeline.py --split test --upper-threshold-percentile 99.9

  # Salvar um HTML 3D por imagem com aorta, óstios e artérias
  python segmentation_pipeline.py --split train --save-segmentation-visuals

  # PROCESSAMENTO EM LOTES (salvamento incremental):
    # Processar em 10 lotes (divide as imagens entre 10 blocos)
    python segmentation_pipeline.py --num-batches 10

    # Processar teste em 5 lotes
    python segmentation_pipeline.py --split test --num-batches 5

  # RETOMADA DE LOTES (em caso de falha):
  # Primeira execução - cria novo diretório
    python segmentation_pipeline.py --split test --num-batches 70
  # Saída: output/segmentation/runs/mid_res/2026-03-14_10-30-00/

  # Se falhar no lote 3, retomar no MESMO diretório:
    python segmentation_pipeline.py --split test --num-batches 70 --resume-batch 3 --resume-dir output/segmentation/runs/mid_res/2026-03-14_10-30-00

    # Retomada explícita por subset:
    python segmentation_pipeline.py --split all --num-batches 70 --resume-batches train=0,val=3,test=0

  # Versão curta (se no mesmo diretório):
    python segmentation_pipeline.py --split test --num-batches 70 --resume-batch 3 --resume-dir ./output/segmentation/runs/mid_res/2026-03-14_10-30-00

  # Apenas consolidar lotes já processados, sem reprocessar imagens:
    python segmentation_pipeline.py --merge-only --split test --resume-dir output/segmentation/runs/mid_res/2026-03-14_10-30-00

  # Sobrescrever o caminho do dataset pela CLI:
    python segmentation_pipeline.py --split test --base-path /dados/ImageCAS/1-1000

Arquivos de saída:
  - numeric/ostios_{split}_summary.csv: Resultados consolidados ao final (ou após merge)
  - numeric/ostios_{split}_lote_1_summary.csv, numeric/ostios_{split}_lote_2_summary.csv, etc: Resultados de cada lote
  - numeric/ostios_{split}_metadata.json: Metadados completos
  - config/effective_pipeline_config.json: Config efetiva usada no run
  - config/split_ids.json: IDs processados por split
  - logs/pipeline.log: Log da execução
  - visual/{split}/*.html: Visualizações 3D quando --save-segmentation-visuals for usado
"""


def parse_resume_batches(resume_batches_arg):
    """Converte um argumento no formato 'train=1,val=0,test=3' em um dicionário."""
    valid_splits = {"train", "val", "test"}
    resume_map = {}

    if not resume_batches_arg:
        return resume_map

    entries = [
        entry.strip() for entry in resume_batches_arg.split(",") if entry.strip()
    ]
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                "Formato inválido para --resume-batches. Use algo como 'train=1,val=0,test=3'."
            )

        split_name, batch_text = entry.split("=", 1)
        split_name = split_name.strip()
        batch_text = batch_text.strip()

        if split_name not in valid_splits:
            raise ValueError(
                f"Split inválido em --resume-batches: {split_name}. Use train, val ou test."
            )

        try:
            batch_num = int(batch_text)
        except ValueError as exc:
            raise ValueError(
                f"Valor inválido para o split '{split_name}' em --resume-batches: {batch_text}"
            ) from exc
        if batch_num < 0:
            raise ValueError(
                f"Valor inválido para o split '{split_name}' em --resume-batches: {batch_text}. Use 0 ou maior."
            )
        resume_map[split_name] = batch_num

    return resume_map


def build_parser(default_base_path, default_output_dir):
    """Cria o parser de argumentos do pipeline."""
    parser = argparse.ArgumentParser(
        description="Pipeline de segmentação coronária",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=PIPELINE_EPILOG,
    )

    parser.add_argument(
        "--split",
        nargs="+",
        choices=["train", "val", "test", "all"],
        default=["all"],
        help="Conjunto(s) para processar (padrão: all)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=["mid", "high"],
        default="mid",
        help="Resolução da imagem: 'mid' (downscale 2x) ou 'high' (sem downscale) (padrão: mid)",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu",
        dest="use_gpu",
        action="store_true",
        default=None,
        help="Força uso de GPU nas etapas compatíveis, se disponível.",
    )
    gpu_group.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="Força CPU nas etapas compatíveis, mesmo se houver GPU disponível.",
    )
    parser.add_argument(
        "--artery-segmentation-method",
        "--artery-method",
        choices=["rg", "fc", "region_growing", "fuzzy_connectedness"],
        default=None,
        help=(
            "Método de segmentação arterial: 'rg'/'region_growing' ou "
            "'fc'/'fuzzy_connectedness'."
        ),
    )
    parser.add_argument(
        "--aorta-ostia-method",
        choices=[
            "standard",
            "bilateral_thin",
            "bilateral_thin_conditional",
        ],
        default=None,
        help=(
            "Método conjunto de aorta/óstios: 'standard' mantém o pipeline "
            "histórico; 'bilateral_thin' ativa superfície fina, seleção "
            "bilateral e correção condicional da máscara da aorta."
        ),
    )
    parser.add_argument(
        "--aorta-level-set-mode",
        choices=["fixed", "adaptive"],
        default=None,
        help=(
            "Controle de iterações do level set da aorta: 'fixed' preserva o "
            "total configurado; 'adaptive' classifica checkpoints e pode "
            "reiniciar uma evolução conservadora ou permissiva a partir de "
            "um checkpoint anterior."
        ),
    )
    parser.add_argument(
        "--aorta-circle-filter",
        choices=["none", "robust"],
        default=None,
        help=(
            "Filtro experimental da trajetória da Hough. 'robust' interpola "
            "outliers isolados e remove uma cauda geometricamente incompatível; "
            "'none' preserva todos os círculos detectados."
        ),
    )
    parser.add_argument(
        "--aorta-circle-filter-min-coverage",
        type=float,
        default=None,
        help=(
            "Cobertura mínima da trajetória para permitir remoção de cauda. "
            "Use 0.4 para reproduzir o filtro agressivo experimental."
        ),
    )
    circle_interpolation_group = parser.add_mutually_exclusive_group()
    circle_interpolation_group.add_argument(
        "--aorta-circle-filter-interpolate",
        dest="aorta_circle_filter_interpolate",
        action="store_true",
        default=None,
        help="Interpola outliers circulares isolados no filtro robusto.",
    )
    circle_interpolation_group.add_argument(
        "--no-aorta-circle-filter-interpolate",
        dest="aorta_circle_filter_interpolate",
        action="store_false",
        help="Desativa a interpolação de outliers circulares isolados.",
    )
    circle_fallback_group = parser.add_mutually_exclusive_group()
    circle_fallback_group.add_argument(
        "--aorta-circle-filter-reject-oversegmented",
        dest="aorta_circle_filter_reject_oversegmented",
        action="store_true",
        default=None,
        help=(
            "Rejeita a trajetória filtrada quando a máscara resultante ainda "
            "for classificada como sobresegmentada e repete somente o level set "
            "com os círculos originais."
        ),
    )
    circle_fallback_group.add_argument(
        "--no-aorta-circle-filter-reject-oversegmented",
        dest="aorta_circle_filter_reject_oversegmented",
        action="store_false",
        help="Desativa o fallback para a trajetória original.",
    )
    parser.add_argument(
        "--ostia-surface-mode",
        choices=["erosion", "physical_distance"],
        default=None,
        help=(
            "Superfície candidata dos óstios: casca por erosão ou faixa "
            "interna definida em distância física."
        ),
    )
    parser.add_argument(
        "--ostia-surface-thickness-mm",
        type=float,
        default=None,
        help="Espessura interna, em mm, da superfície candidata dos óstios.",
    )
    parser.add_argument(
        "--ostia-candidate-score-mode",
        choices=["voxel", "local_mean", "external_mean", "robust_percentile"],
        default=None,
        help="Score usado para ordenar candidatos de óstio.",
    )
    parser.add_argument(
        "--ostia-pair-selection-mode",
        choices=["greedy", "joint", "bilateral"],
        default=None,
        help="Estratégia para selecionar o par de óstios.",
    )
    parser.add_argument(
        "--aorta-leak-correction",
        choices=[
            "none",
            "circle_seeded_neck_pruning",
            "circle_area_jump_pruning",
        ],
        default=None,
        help=(
            "Correção experimental posterior ao level set. "
            "'circle_seeded_neck_pruning' tenta remover vazamentos inferiores "
            "conectados por colos estreitos; 'circle_area_jump_pruning' busca "
            "uma expansão abrupta da área axial. Ambas requerem adaptive."
        ),
    )
    parser.add_argument(
        "--aorta-neck-pruning-erosion-radius",
        type=int,
        default=None,
        help=(
            "Raio da erosão 3D usada pela poda experimental de colos estreitos. "
            "Valores 3 e 4 são mais agressivos que o padrão 2."
        ),
    )
    parser.add_argument(
        "--aorta-neck-pruning-core-radius-factor",
        type=float,
        default=None,
        help=(
            "Fração do raio dos círculos preservada como núcleo da aorta pela "
            "poda experimental."
        ),
    )
    parser.add_argument(
        "--aorta-neck-pruning-max-volume-loss",
        type=float,
        default=None,
        help=(
            "Perda relativa máxima de volume aceita pela poda experimental, "
            "por exemplo 0.15 para 15%%."
        ),
    )
    parser.add_argument(
        "--rg-comparison-window",
        type=int,
        default=None,
        help=(
            "Referência de comparação do region growing: 1 compara com o voxel "
            "atual, -1 compara com a média acumulada da região, valores >1 "
            "comparam com a média dos últimos N voxels aceitos."
        ),
    )
    parser.add_argument(
        "--threshold-method",
        choices=["normal", "fuzzy"],
        default=None,
        help=(
            "Threshold inicial: 'normal' usa o piso configurado até percentil; "
            "'fuzzy' mantém voxels cuja maior pertinência é objeto."
        ),
    )
    parser.add_argument(
        "--lower-threshold-method",
        choices=["fixed", "percentile"],
        default=None,
        help=(
            "Método do limiar inferior HU: fixed usa MIN_THRESHOLD; "
            "percentile usa percentil baixo dos voxels válidos."
        ),
    )
    parser.add_argument(
        "--lower-threshold-percentile",
        type=float,
        default=None,
        help="Percentil baixo usado nos métodos adaptativos de limiar inferior.",
    )
    parser.add_argument(
        "--upper-threshold-percentile",
        type=float,
        default=None,
        help="Percentil superior usado pelo threshold normal.",
    )
    parser.add_argument(
        "--lower-threshold-clip-min",
        type=float,
        default=None,
        help="HU mínimo da faixa considerada no cálculo adaptativo do piso.",
    )
    parser.add_argument(
        "--lower-threshold-clip-max",
        type=float,
        default=None,
        help="HU máximo da faixa considerada no cálculo adaptativo do piso.",
    )
    parser.add_argument(
        "--downscale-method",
        type=str,
        choices=["scipy", "opencv"],
        default=None,
        help="Método de downscaling: scipy (ndi.zoom) ou opencv (cv2.resize)",
    )
    parser.add_argument(
        "--opencv-interpolation",
        type=str,
        choices=["nearest", "linear", "cubic", "area", "lanczos4"],
        default=None,
        help="Método de interpolação do OpenCV (usado apenas se --downscale-method=opencv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help=f"Diretório de saída (padrão: {default_output_dir})",
    )
    parser.add_argument(
        "--run-group",
        type=str,
        default=None,
        help=(
            "Subpasta relativa dentro de runs/<resolução>_res para organizar "
            "a execução, por exemplo aorta_segmentation_experiments/train/variant."
        ),
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=str(default_base_path),
        help=(
            "Diretório base do dataset ImageCAS "
            f"(padrão: {default_base_path}; se indisponível, o pipeline tenta o fallback configurado)"
        ),
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Apenas mescla CSVs de lotes existentes e atualiza metadados. Use com --resume-dir.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Arquivo JSON com configurações para sobrescrever valores padrão",
    )
    parser.add_argument(
        "--split-config",
        type=str,
        default=None,
        help=(
            "Arquivo JSON alternativo com splits train/val/test. "
            "Útil para testar tamanhos diferentes de treino sem alterar config/imagecas_splits.json."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Habilitar logging detalhado (DEBUG)",
    )
    parser.add_argument(
        "--save-segmentation-visuals",
        action="store_true",
        help=(
            "Salva um HTML 3D interativo por imagem com aorta, óstios, "
            "artéria predita e ground truth."
        ),
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Número de lotes para dividir o conjunto de imagens (ex: 5 divide as 700 imagens em 5 lotes)",
    )
    parser.add_argument(
        "--resume-batch",
        type=int,
        default=0,
        help="Número do lote para retomar, em numeração humana/arquivo (ex: 11 retoma salvando lote_11). Padrão: 0 = começar do início.",
    )
    parser.add_argument(
        "--resume-batches",
        type=str,
        default=None,
        help="Retomada explícita por subset no formato 'train=1,val=0,test=3'. Se informado, sobrescreve --resume-batch para os splits listados.",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Diretório anterior para retomar (ex: output/segmentation/runs/mid_res/2026-03-14_10-30-00). Obrigatório quando a retomada começa de um lote > 0.",
    )
    return parser


def parse_pipeline_args(default_base_path, default_output_dir):
    """Parseia, valida e normaliza argumentos do pipeline."""
    parser = build_parser(default_base_path, default_output_dir)
    args = parser.parse_args()

    if args.resume_batch < 0:
        parser.error("--resume-batch deve ser 0 ou maior")
    if args.num_batches <= 0:
        parser.error("--num-batches deve ser maior que 0")
    if args.merge_only and not args.resume_dir:
        parser.error("--merge-only requer --resume-dir com a pasta de saída existente")
    if args.run_group:
        run_group = Path(args.run_group)
        if run_group.is_absolute() or ".." in run_group.parts:
            parser.error("--run-group deve ser um caminho relativo sem '..'")

    try:
        resume_batches_overrides = parse_resume_batches(args.resume_batches)
    except ValueError as exc:
        parser.error(str(exc))

    args.base_path = Path(args.base_path).expanduser()
    args.output_dir = Path(args.output_dir).expanduser()
    args.resume_dir = Path(args.resume_dir).expanduser() if args.resume_dir else None
    args.split_config = (
        Path(args.split_config).expanduser() if args.split_config else None
    )
    args.resume_batches_by_split = {
        "train": args.resume_batch,
        "val": args.resume_batch,
        "test": args.resume_batch,
    }
    args.resume_batches_by_split.update(resume_batches_overrides)
    args.resume_requested = any(
        batch > 0 for batch in args.resume_batches_by_split.values()
    )

    return args
