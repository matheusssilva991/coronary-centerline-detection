"""Argumentos de linha de comando para o pipeline de segmentação."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_rg_comparison_window(value: str) -> int:
    """Converte ``ALL`` ou um tamanho inteiro para a representação interna."""
    normalized = value.strip().upper()
    if normalized == "ALL":
        return -1

    try:
        window = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use ALL, -1 ou um número inteiro positivo"
        ) from exc

    if window == 0 or window < -1:
        raise argparse.ArgumentTypeError(
            "use ALL, -1 ou um número inteiro positivo"
        )
    return window


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

  # Ativar o controle adaptativo das iterações do level set da aorta
  python segmentation_pipeline.py --split train --aorta-level-set-mode adaptive

  # Comparar candidatos do region growing com a média acumulada da região
  python segmentation_pipeline.py --split train --rg-comparison-window ALL

  # Usar fuzzy threshold
  python segmentation_pipeline.py --split val --threshold-method fuzzy --lower-threshold-percentile 10.5

  # Testar limiar inferior adaptativo mantendo threshold normal + RG
  python segmentation_pipeline.py --split train --threshold-method normal --artery-method rg --lower-threshold-method percentile --lower-threshold-percentile 10.75

  # Sobrescrever apenas o percentil superior do threshold
  python segmentation_pipeline.py --split test --upper-threshold-percentile 99.9

  # Salvar um HTML 3D por imagem com aorta, óstios e artérias
  python segmentation_pipeline.py --split train --save-segmentation-visuals

  # Salvar os HTMLs em um disco externo, mantendo CSVs e logs no repositório
  python segmentation_pipeline.py --split train --save-segmentation-visuals --visual-output-dir /media/matheus/HD/ImageCAS_pipeline_results

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
  - visual/{split}/*.html: Visualizações 3D; pode ser redirecionado com --visual-output-dir
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
        "--aorta-level-set-mode",
        choices=["fixed", "adaptive"],
        default=None,
        help=(
            "Controle de iterações do level set da aorta: 'fixed' preserva o "
            "total configurado; 'adaptive' classifica checkpoints e pode "
            "reiniciar uma evolução conservadora a partir de um checkpoint "
            "anterior quando R_P90 indicar sobresegmentação."
        ),
    )
    parser.add_argument(
        "--aorta-level-set-iterations",
        type=int,
        default=None,
        help="Número de iterações da evolução nominal do level set da aorta.",
    )
    parser.add_argument(
        "--aorta-level-set-radius-reduction-factor",
        type=float,
        default=None,
        help=(
            "Fração do raio de cada círculo usada para inicializar o level set. "
            "Por exemplo, 0.15 cria sementes com 15%% do raio detectado."
        ),
    )
    parser.add_argument(
        "--aorta-level-set-balloon",
        type=float,
        default=None,
        help=(
            "Força balloon da evolução nominal da aorta. Valores menores "
            "tornam a expansão mais conservadora."
        ),
    )
    parser.add_argument(
        "--aorta-level-set-alpha",
        type=float,
        default=None,
        help=(
            "Sensibilidade do mapa de bordas do level set da aorta. Valores "
            "maiores reforçam a influência das bordas."
        ),
    )
    parser.add_argument(
        "--aorta-opening-radius",
        type=int,
        default=None,
        help=(
            "Raio, em voxels, da abertura morfológica aplicada à máscara da "
            "aorta após o level set. Use 0 para desativar a abertura."
        ),
    )
    parser.add_argument(
        "--aorta-trajectory-radius-factor",
        type=float,
        default=None,
        help=(
            "Restringe a máscara pós-level set ao envelope dos círculos com "
            "raio k*r. Use somente para experimentos; valores sugeridos: "
            "1.75, 2.0 e 2.25."
        ),
    )
    parser.add_argument(
        "--aorta-trajectory-axial-margin-slices",
        type=int,
        default=None,
        help=(
            "Prolonga o envelope da trajetória antes do primeiro e depois do "
            "último círculo. Use com --aorta-trajectory-radius-factor; valor "
            "experimental sugerido: 5 fatias."
        ),
    )
    parser.add_argument(
        "--aorta-oversegmented-area-ratio-p90",
        type=float,
        default=None,
        help=(
            "Sobrescreve o limiar de R_P90 que aciona a evolução conservadora "
            "do level set adaptativo."
        ),
    )
    parser.add_argument(
        "--aorta-conservative-balloon",
        type=float,
        default=None,
        help="Forca balloon usada na evolucao conservadora da aorta.",
    )
    parser.add_argument(
        "--aorta-conservative-alpha",
        type=float,
        default=None,
        help="Sensibilidade ao mapa de bordas na evolucao conservadora.",
    )
    parser.add_argument(
        "--aorta-conservative-threshold-percentile",
        type=float,
        default=None,
        help=(
            "Percentil do mapa de gradiente usado como threshold na evolucao "
            "conservadora. Deve estar entre 0 e 100."
        ),
    )
    parser.add_argument(
        "--aorta-conservative-min-ratio-improvement",
        type=float,
        default=None,
        help=(
            "Reducao relativa minima de R_P90 exigida para aceitar a evolucao "
            "conservadora. Deve estar entre 0 e 1."
        ),
    )
    localization_leak_override_group = parser.add_mutually_exclusive_group()
    localization_leak_override_group.add_argument(
        "--aorta-localization-leak-override",
        dest="aorta_localization_leak_override",
        action="store_true",
        default=None,
        help=(
            "Permite tentar a evolucao conservadora quando a localizacao dos "
            "circulos e suspeita, mas R_P90, preenchimento e volume indicam "
            "conjuntamente um vazamento forte."
        ),
    )
    localization_leak_override_group.add_argument(
        "--no-aorta-localization-leak-override",
        dest="aorta_localization_leak_override",
        action="store_false",
        help="Desativa a excecao de vazamento para localizacoes suspeitas.",
    )
    parser.add_argument(
        "--aorta-localization-leak-min-area-ratio-p90",
        type=float,
        default=None,
        help="R_P90 minimo, exclusivo, para superar o bloqueio por localizacao.",
    )
    parser.add_argument(
        "--aorta-localization-leak-min-circle-fill-q25",
        type=float,
        default=None,
        help="Preenchimento Q25 minimo dos circulos para permitir a excecao.",
    )
    parser.add_argument(
        "--aorta-localization-leak-min-volume-fraction",
        type=float,
        default=None,
        help="Fracao volumetrica minima da aorta para permitir a excecao.",
    )
    parser.add_argument(
        "--aorta-circle-filter",
        choices=["none", "robust"],
        default=None,
        help=(
            "Filtro experimental da trajetória da Hough. 'robust' remove uma "
            "cauda geometricamente incompatível e pode extrapolar uma continuação "
            "curta a partir da última região estável; "
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
    parser.add_argument(
        "--aorta-circle-filter-max-trim-fraction",
        type=float,
        default=None,
        help=(
            "Fração máxima da trajetória original que o filtro robusto pode "
            "remover. Use 0.4 para rejeitar cortes axiais maiores que 40%%."
        ),
    )
    parser.add_argument(
        "--aorta-circle-filter-synthetic-tail-slices",
        type=int,
        default=None,
        help=(
            "Número de fatias sintéticas extrapoladas da última região estável "
            "depois que uma cauda incompatível é removida."
        ),
    )
    mask_guided_group = parser.add_mutually_exclusive_group()
    mask_guided_group.add_argument(
        "--aorta-circle-filter-mask-guided",
        dest="aorta_circle_filter_mask_guided",
        action="store_true",
        default=None,
        help=(
            "Ativa o fallback experimental que usa R_z da máscara nominal "
            "para detectar e substituir uma cauda circular com vazamento."
        ),
    )
    mask_guided_group.add_argument(
        "--no-aorta-circle-filter-mask-guided",
        dest="aorta_circle_filter_mask_guided",
        action="store_false",
        help="Desativa o fallback guiado pela máscara nominal.",
    )
    parser.add_argument(
        "--aorta-mask-guided-area-ratio-p90",
        type=float,
        default=None,
        help=(
            "Limiar de R_P90 que permite tentar e aceitar o fallback guiado "
            "pela máscara."
        ),
    )
    parser.add_argument(
        "--aorta-mask-guided-max-fill-loss",
        type=float,
        default=None,
        help="Perda máxima permitida em circle_fill_q25 no fallback guiado.",
    )
    parser.add_argument(
        "--aorta-mask-guided-min-ratio-improvement",
        type=float,
        default=None,
        help="Redução relativa mínima exigida em R_P90 para aceitar o fallback.",
    )
    parser.add_argument(
        "--rg-comparison-window",
        type=_parse_rg_comparison_window,
        default=None,
        help=(
            "Referência de comparação do region growing: 1 compara com o voxel "
            "atual; ALL (ou -1) compara com a média acumulada de todos os voxels "
            "aceitos; valores >1 comparam com a média dos últimos N voxels."
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
        "--image-ids",
        type=str,
        default=None,
        help=(
            "Processa somente os IDs informados, separados por vírgula, dentro "
            "dos splits selecionados. Exemplo: --image-ids 44,330,603."
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
        "--visual-output-dir",
        type=str,
        default=None,
        help=(
            "Raiz alternativa para os HTMLs 3D. A estrutura relativa do run "
            "é preservada nessa raiz; CSVs, configs e logs continuam em --output-dir."
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
    if (
        args.aorta_trajectory_radius_factor is not None
        and args.aorta_trajectory_radius_factor <= 0
    ):
        parser.error("--aorta-trajectory-radius-factor deve ser maior que 0")
    if (
        args.aorta_level_set_iterations is not None
        and args.aorta_level_set_iterations <= 0
    ):
        parser.error("--aorta-level-set-iterations deve ser maior que 0")
    if (
        args.aorta_level_set_radius_reduction_factor is not None
        and args.aorta_level_set_radius_reduction_factor <= 0
    ):
        parser.error(
            "--aorta-level-set-radius-reduction-factor deve ser maior que 0"
        )
    if args.aorta_level_set_alpha is not None and args.aorta_level_set_alpha <= 0:
        parser.error("--aorta-level-set-alpha deve ser maior que 0")
    if args.aorta_opening_radius is not None and args.aorta_opening_radius < 0:
        parser.error("--aorta-opening-radius deve ser zero ou maior")
    if (
        args.aorta_trajectory_axial_margin_slices is not None
        and args.aorta_trajectory_axial_margin_slices < 0
    ):
        parser.error(
            "--aorta-trajectory-axial-margin-slices deve ser zero ou maior"
        )
    if (
        args.aorta_oversegmented_area_ratio_p90 is not None
        and args.aorta_oversegmented_area_ratio_p90 <= 0
    ):
        parser.error("--aorta-oversegmented-area-ratio-p90 deve ser maior que 0")
    if (
        args.aorta_conservative_balloon is not None
        and args.aorta_conservative_balloon <= 0
    ):
        parser.error("--aorta-conservative-balloon deve ser maior que 0")
    if (
        args.aorta_conservative_alpha is not None
        and args.aorta_conservative_alpha <= 0
    ):
        parser.error("--aorta-conservative-alpha deve ser maior que 0")
    if (
        args.aorta_conservative_threshold_percentile is not None
        and not 0 <= args.aorta_conservative_threshold_percentile <= 100
    ):
        parser.error(
            "--aorta-conservative-threshold-percentile deve estar entre 0 e 100"
        )
    if (
        args.aorta_conservative_min_ratio_improvement is not None
        and not 0 <= args.aorta_conservative_min_ratio_improvement <= 1
    ):
        parser.error(
            "--aorta-conservative-min-ratio-improvement deve estar entre 0 e 1"
        )
    if (
        args.aorta_localization_leak_min_area_ratio_p90 is not None
        and args.aorta_localization_leak_min_area_ratio_p90 <= 0
    ):
        parser.error(
            "--aorta-localization-leak-min-area-ratio-p90 deve ser maior que 0"
        )
    if (
        args.aorta_localization_leak_min_circle_fill_q25 is not None
        and not 0 <= args.aorta_localization_leak_min_circle_fill_q25 <= 1
    ):
        parser.error(
            "--aorta-localization-leak-min-circle-fill-q25 deve estar entre 0 e 1"
        )
    if (
        args.aorta_localization_leak_min_volume_fraction is not None
        and not 0 <= args.aorta_localization_leak_min_volume_fraction <= 1
    ):
        parser.error(
            "--aorta-localization-leak-min-volume-fraction deve estar entre 0 e 1"
        )
    if (
        args.aorta_mask_guided_area_ratio_p90 is not None
        and args.aorta_mask_guided_area_ratio_p90 <= 0
    ):
        parser.error("--aorta-mask-guided-area-ratio-p90 deve ser maior que 0")
    for option, value in (
        ("--aorta-mask-guided-max-fill-loss", args.aorta_mask_guided_max_fill_loss),
        (
            "--aorta-mask-guided-min-ratio-improvement",
            args.aorta_mask_guided_min_ratio_improvement,
        ),
    ):
        if value is not None and not 0 <= value <= 1:
            parser.error(f"{option} deve estar entre 0 e 1")
    if args.image_ids:
        try:
            args.image_ids = [
                int(value.strip())
                for value in args.image_ids.split(",")
                if value.strip()
            ]
        except ValueError:
            parser.error("--image-ids deve conter somente inteiros separados por vírgula")
        if not args.image_ids:
            parser.error("--image-ids não pode ser vazio")
        args.image_ids = list(dict.fromkeys(args.image_ids))
    if args.merge_only and not args.resume_dir:
        parser.error("--merge-only requer --resume-dir com a pasta de saída existente")
    if args.merge_only and args.image_ids:
        parser.error("--image-ids não pode ser usado com --merge-only")
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
    args.visual_output_dir = (
        Path(args.visual_output_dir).expanduser() if args.visual_output_dir else None
    )
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
