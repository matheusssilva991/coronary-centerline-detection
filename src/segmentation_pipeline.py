"""CLI principal do pipeline ImageCAS: seleção de split, lotes, consolidação e metadados."""

import copy
import json
import logging
import os
from pathlib import Path

import pandas as pd

# Usa GPU 0 por padrão quando a variável não for definida externamente.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from utils.processing.gpu_utils import use_gpu
from utils.project.config import load_config_json
from utils.project.datasets.imagecas import get_data_splits, list_dataset_image_ids
from utils.project.results import (
    ResultIntegrityError,
    batch_result_number,
    create_timestamped_output_dir,
    list_batch_result_files,
    load_batch_timing_records,
    merge_batch_results,
    save_metadata,
    summarize_batch_timing_records,
    validate_result_integrity,
)
from utils.project.results.artery_masks import ArteryMaskExport
from utils.project.results.io import (
    integrity_filename,
    metadata_candidates,
    metadata_filename,
)
from utils.project.runtime.run_reporting import (
    add_run_file_handler,
    create_standard_run_dirs,
    notify_run_completion,
    notify_run_exception,
)
from utils.segmentation.pipeline.cli import (
    _apply_execution_overrides as _apply_execution_overrides,
)
from utils.segmentation.pipeline.cli import (
    _apply_threshold_overrides as _apply_threshold_overrides,
)
from utils.segmentation.pipeline.cli import (
    build_effective_config as _build_effective_config,
)
from utils.segmentation.pipeline.cli import parse_pipeline_args
from utils.segmentation.pipeline.cli import (
    select_resolution_config as _select_resolution_config,
)
from utils.segmentation.pipeline.orchestration import run_pipeline
from utils.segmentation.pipeline.reporting import print_split_summary, print_statistics
from utils.utils.json_io import save_json_atomic

logger = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

GPU_ENABLED = use_gpu()
if GPU_ENABLED:
    logger.info("GPU detectada! Operações aceleradas por GPU ativadas.")
else:
    logger.warning("GPU não disponível. Acelerações CPU usadas.")

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = Path("/run/media/matheus/HD/DatasetsCCTA/ImageCAS/1-1000")
BASE_PATH_FALLBACK = Path("/data04/home/mpmaia/ImageCAS/database/1-1000")
OUTPUT_DIR = REPO_ROOT / "output"
PIPELINE_CONFIG_PATH = REPO_ROOT / "config" / "pipeline_config.json"


def has_imagecas_images(path: Path) -> bool:
    """Verifica se o diretório contém exames ImageCAS no padrão ``*.img.nii.gz``.

    Args:
        path (Path): Caminho do diretório candidato a raiz do dataset.

    Returns:
        bool: ``True`` se ``path`` existir e contiver ao menos um arquivo
            ``*.img.nii.gz``.
    """
    return path.exists() and any(path.glob("*.img.nii.gz"))


def resolve_base_path(base_path: Path) -> Path:
    """Resolve o caminho do ImageCAS usando fallback apenas para o caminho padrão.

    Args:
        base_path (Path): Diretório informado via CLI ou default local ``BASE_PATH``.

    Returns:
        Path: ``base_path`` quando válido ou ``BASE_PATH_FALLBACK`` se o padrão local
            não existir.
    """
    # Prioriza o diretório informado quando já contém exames válidos.
    if has_imagecas_images(base_path):
        return base_path

    # Recorre ao caminho alternativo somente quando o default local não existe.
    if base_path == BASE_PATH and has_imagecas_images(BASE_PATH_FALLBACK):
        print(
            "⚠️  Dataset não encontrado no caminho padrão local. "
            f"Usando fallback: {BASE_PATH_FALLBACK}"
        )
        logger.info(
            "Dataset não encontrado em %s; usando fallback %s",
            base_path,
            BASE_PATH_FALLBACK,
        )
        return BASE_PATH_FALLBACK

    return base_path


def load_default_config(config_path=PIPELINE_CONFIG_PATH):
    """Carrega a configuração base do pipeline a partir de JSON (com fallback mínimo).

    Args:
        config_path (Path | str): Caminho do arquivo ``pipeline_config.json`` a
            carregar.

    Returns:
        dict[str, Any]: Dicionário de configuração carregado e normalizado para
            runtime.
    """
    # Carrega a configuração JSON principal com fallback mínimo em caso de falha.
    try:
        config = load_config_json(str(config_path), {})
        logger.info("Config carregada de pipeline_config.json: %s", config_path)
        return config
    except Exception as exc:
        logger.warning(
            "Falha ao carregar %s: %s. Usando defaults mínimos.",
            config_path,
            exc,
        )
        return {
            "USE_GPU": GPU_ENABLED,
            "NUM_BATCHES": 5,
        }


CONFIG = load_default_config()
CONFIG_MID_RES = copy.deepcopy(CONFIG)
CONFIG_HIGH_RES = copy.deepcopy(CONFIG)
CONFIG_HIGH_RES["DOWNSCALE_FACTORS"] = [1, 1, 1]


def select_resolution_config(args):
    """Seleciona a configuração base para mid ou high resolution.

    Args:
        args (argparse.Namespace): Namespace CLI contendo ``resolution``.

    Returns:
        dict[str, Any]: Dicionário ``CONFIG_MID_RES`` ou ``CONFIG_HIGH_RES``.
    """
    return _select_resolution_config(args, CONFIG_MID_RES, CONFIG_HIGH_RES)


def build_effective_config(args):
    """Monta a configuração efetiva aplicando resolução, JSON extra, flags e escala espacial.

    Args:
        args (argparse.Namespace): Namespace CLI validado do pipeline ImageCAS.

    Returns:
        dict[str, Any]: Dicionário de configuração final normalizado e escalado
            para a resolução.
    """
    return _build_effective_config(args, CONFIG_MID_RES, CONFIG_HIGH_RES)


def print_run_settings(args, config, base_path):
    """Exibe no console um resumo das configurações operacionais da execução.

    Args:
        args (argparse.Namespace): Namespace CLI com quantidade de lotes e lote de
            retomada.
        config (dict[str, Any]): Configuração efetiva já resolvida.
        base_path (Path): Diretório efetivo do dataset ImageCAS utilizado.
    """
    # Exibe método de downscale, origem do dataset e disponibilidade de GPU.
    if config["DOWNSCALE_METHOD"] == "opencv":
        print(
            "🔧 Método de downscale: "
            f"{config['DOWNSCALE_METHOD']} "
            f"(interpolação: {config['OPENCV_INTERPOLATION']})"
        )
    else:
        print(f"🔧 Método de downscale: {config['DOWNSCALE_METHOD']}")

    print(f"🗂️  Dataset: {base_path}")
    print(
        "🖥️  GPU nas etapas compatíveis: "
        f"{'habilitada' if config.get('USE_GPU', False) else 'desabilitada'}"
    )
    # Exibe parâmetros da localização da aorta, segmentação arterial e thresholding.
    circle_config = config.get("CIRCLE_DETECTION", {})
    circle_filter_config = circle_config.get("trajectory_filter", {})
    print("🔎 LCC: por fatia")
    print(
        "⭕ Localização da aorta: "
        f"miss_count={circle_config.get('max_slice_miss_threshold')}, "
        "filtro de trajetória="
        f"{circle_filter_config.get('method', 'none')} "
        f"(cobertura mínima={circle_filter_config.get('min_tail_coverage', 0.8)})"
    )
    artery_config = config.get("ARTERY_SEGMENTATION", {})
    thresholding_config = config.get("THRESHOLDING", {})
    lower_threshold_config = config.get("LOWER_THRESHOLD", {})
    print(f"🫀 Segmentação arterial: {artery_config.get('method', 'region_growing')}")
    print("🎯 Superfície/seleção dos óstios: erosion / greedy")
    print("🔄 Level set da aorta: fixed")
    print(f"🧩 Threshold: {thresholding_config.get('method', 'normal')}")
    print(
        "🧱 Piso inferior: "
        f"{lower_threshold_config.get('method', 'fixed')} "
        f"(p={lower_threshold_config.get('percentile', 'N/A')})"
    )
    # Exibe organização em lotes, flag de visuais 3D e eventual lote de retomada.
    print(f"📦 Processamento em {args.num_batches} lotes")
    print(
        "🖼️  Visualizações 3D: "
        f"{'salvas' if config.get('SAVE_SEGMENTATION_VISUALS') else 'desativadas'}"
    )

    if args.resume_batch > 0:
        print(f"🔄 Retomando a partir do lote {args.resume_batch}")


def resolve_output_dir(args, output_root_dir):
    """Cria uma nova pasta de execução com timestamp ou reutiliza ``--resume-dir``.

    Args:
        args (argparse.Namespace): Namespace CLI contendo flags de retomada,
            consolidação, resolução e grupo.
        output_root_dir (Path): Diretório raiz onde novos runs são criados.

    Returns:
        dict[str, Path]: Dicionário com caminhos ``run_dir``, ``numeric_dir``,
            ``config_dir``, ``visual_dir`` e ``logs_dir``.
    """
    # Em retomada ou consolidação (--merge-only), reaproveita o diretório existente do run.
    if (args.resume_requested or args.merge_only) and args.resume_dir:
        if args.resume_dir.exists():
            print(f"\n📁 Usando diretório anterior: {args.resume_dir}\n")
            output_dirs = resolve_existing_output_dirs(args.resume_dir)
            output_dirs["visual_dir"] = resolve_visual_output_dir(
                args,
                output_dirs["run_dir"],
                output_root_dir,
            )
            return output_dirs
        print(f"❌ Erro: Diretório não encontrado: {args.resume_dir}")
        print("   Use --resume-dir com o caminho do diretório anterior")
        raise SystemExit(1)

    if args.resume_requested:
        print("❌ Erro: para retomar a partir de um lote, informe --resume-dir")
        print(
            "   Exemplo: --resume-batch 11 "
            "--resume-dir output/segmentation/2026-05-19_17-08-33"
        )
        raise SystemExit(1)

    # Nova execução: cria pasta com timestamp separada por resolução e grupo opcional.
    experiment_name = f"segmentation/runs/{args.resolution}_res"
    if args.run_group:
        experiment_name = f"{experiment_name}/{Path(args.run_group).as_posix()}"
    run_dir = create_timestamped_output_dir(
        output_root_dir,
        experiment_name=experiment_name,
    )
    output_dirs = build_structured_output_dirs(run_dir)
    output_dirs["visual_dir"] = resolve_visual_output_dir(
        args,
        output_dirs["run_dir"],
        output_root_dir,
    )
    print(f"📁 Diretório da execução: {output_dirs['run_dir']}")
    print(f"📊 Resultados numéricos: {output_dirs['numeric_dir']}")
    print(f"🖼️  Exemplos visuais: {output_dirs['visual_dir']} (criada sob demanda)\n")
    return output_dirs


def resolve_visual_output_dir(args, run_dir, output_root_dir):
    """Resolve o diretório visual dentro do run ou espelhado em ``--visual-output-dir``.

    Args:
        args (argparse.Namespace): Namespace CLI contendo ``visual_output_dir`` e
            ``resolution``.
        run_dir (Path | str): Diretório raiz do run atual.
        output_root_dir (Path | str): Raiz base de saída usada para calcular o
            caminho relativo espelhado.

    Returns:
        Path: Caminho ``Path`` onde as visualizações HTML do run devem ser salvas.
    """
    visual_root = getattr(args, "visual_output_dir", None)
    if visual_root is None:
        return Path(run_dir) / "visual"

    run_path = Path(run_dir)
    try:
        # Preserva segmentation/runs/<resolução>/<grupo>/<timestamp> no disco externo.
        relative_run = run_path.resolve().relative_to(Path(output_root_dir).resolve())
    except ValueError:
        # Runs retomados fora de --output-dir ainda recebem um caminho estável.
        relative_run = (
            Path("segmentation") / "runs" / f"{args.resolution}_res" / run_path.name
        )
    return Path(visual_root) / relative_run / "visual"


def build_structured_output_dirs(run_dir):
    """Cria a estrutura padrão de subdiretórios de uma execução nova.

    Args:
        run_dir (Path | str): Diretório raiz do run onde ``numeric``, ``config``,
            ``visual`` e ``logs`` são criados.

    Returns:
        dict[str, Path]: Dicionário mapeando cada chave de subdiretório ao seu
            ``Path``.
    """
    return create_standard_run_dirs(run_dir)


def resolve_existing_output_dirs(resume_dir):
    """Resolve diretórios para retomada a partir do layout novo (``numeric/``) ou legado.

    Args:
        resume_dir (Path | str): Caminho informado pelo usuário como diretório do
            run anterior.

    Returns:
        dict[str, Path]: Dicionário com caminhos ``run_dir``, ``numeric_dir``,
            ``config_dir``, ``visual_dir`` e ``logs_dir``.
    """
    # Aceita a raiz do run, a subpasta numeric ou o layout legado plano.
    resume_dir = Path(resume_dir)
    if resume_dir.name == "numeric":
        run_dir = resume_dir.parent
        numeric_dir = resume_dir
    elif (resume_dir / "numeric").exists():
        run_dir = resume_dir
        numeric_dir = resume_dir / "numeric"
    else:
        run_dir = resume_dir
        numeric_dir = resume_dir

    # Monta o mapa padrão de diretórios e garante as pastas obrigatórias de escrita.
    output_dirs = {
        "run_dir": run_dir,
        "numeric_dir": numeric_dir,
        "config_dir": run_dir / "config",
        "visual_dir": run_dir / "visual",
        "logs_dir": run_dir / "logs",
    }
    output_dirs["numeric_dir"].mkdir(parents=True, exist_ok=True)
    output_dirs["logs_dir"].mkdir(parents=True, exist_ok=True)
    return output_dirs


def setup_file_logging(logs_dir):
    """Configura um arquivo ``pipeline.log`` dentro do diretório de logs da execução.

    Args:
        logs_dir (Path | str): Diretório onde o arquivo ``pipeline.log`` será
            criado.
    """
    try:
        fh_path = Path(logs_dir) / "pipeline.log"
        add_run_file_handler(
            fh_path, formatter=logging.Formatter(LOG_FORMAT), level=logging.DEBUG
        )
        logger.info("Logs também serão gravados em: %s", fh_path)
    except Exception:
        logger.warning("Não foi possível criar arquivo de log no diretório de saída.")


def save_run_snapshots(
    output_dirs,
    config,
    split_name,
    image_ids,
    split_config_path=None,
):
    """Salva snapshots JSON com a configuração efetiva e os IDs da coorte executada.

    Args:
        output_dirs (dict[str, Path]): Dicionário de caminhos do run contendo
            ``config_dir``.
        config (dict[str, Any]): Configuração efetiva normalizada e escalada da
            execução.
        split_name (str): Nome do split executado (``train``, ``val``, ``test`` ou
            ``full``).
        image_ids (list[int] | None): Lista ordenada de IDs da coorte selecionada
            (ou ``None`` para omitir).
        split_config_path (Path | str | None): Caminho opcional do arquivo
            ``split_config`` registrado na origem.
    """
    # Garante a pasta de configuração e salva a configuração efetiva de forma atômica.
    config_dir = output_dirs["config_dir"]
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "effective_pipeline_config.json"
    save_json_atomic(config, config_path)

    # Persiste a lista exata de IDs do split para reprodutibilidade e --merge-only.
    if image_ids is not None:
        split_payload = {
            "source": split_config_path,
            "splits": {split_name: image_ids},
        }
        split_path = config_dir / "split_ids.json"
        save_json_atomic(split_payload, split_path)


def build_split_to_run(args, base_path):
    """Resolve o nome da coorte solicitada e a lista ordenada de IDs a processar.

    Args:
        args (argparse.Namespace): Namespace CLI com ``split``, ``split_config``,
            ``image_ids`` e ``merge_only``.
        base_path (Path): Diretório raiz do dataset ImageCAS.

    Returns:
        tuple[str, list[int] | None]: Tupla ``(split_name, image_ids)``
            (``image_ids`` é ``None`` no modo ``--merge-only``).
    """
    # --merge-only obtém os IDs diretamente dos snapshots do run, sem ler o dataset.
    if args.merge_only:
        return args.split, None

    # Resolve todos os IDs do split configurado antes de aplicar eventual filtro explícito.
    if args.split == "full":
        selected_ids = list_dataset_image_ids(base_path)
        print(
            f"📊 Total: {len(selected_ids)} imagens "
            "(modo full, sem divisão train/val/test)"
        )
    else:
        train_ids, val_ids, test_ids, all_ids = get_data_splits(
            base_path,
            split_config_path=args.split_config,
        )
        print_statistics(train_ids, val_ids, test_ids, all_ids)
        split_map = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }
        selected_ids = split_map[args.split]

    # Filtra apenas os IDs explicitados por --image-ids mantendo a ordem do split.
    requested_ids = getattr(args, "image_ids", None)
    if not requested_ids:
        return args.split, selected_ids

    selected_id_set = {int(img_id) for img_id in selected_ids}
    missing_ids = sorted(set(requested_ids).difference(selected_id_set))
    if missing_ids:
        raise ValueError(
            f"IDs de --image-ids não pertencem ao split {args.split!r}: {missing_ids}"
        )

    requested_id_set = set(requested_ids)
    filtered_ids = [
        img_id for img_id in selected_ids if int(img_id) in requested_id_set
    ]
    print(f"🎯 Seleção explícita: {len(filtered_ids)} imagens (--image-ids)")
    return args.split, filtered_ids


def load_merge_only_image_ids(output_dirs, split_name):
    """Carrega os IDs da coorte persistida no run sem consultar o dataset ImageCAS.

    Args:
        output_dirs (dict[str, Path]): Dicionário de diretórios do run contendo
            ``config_dir`` e ``numeric_dir``.
        split_name (str): Nome do split cujo snapshot ``split_ids.json`` ou
            metadata será lido.

    Returns:
        list[int]: Lista de IDs inteiros da coorte original do run.
    """
    # Prioriza o snapshot split_ids.json gravado nos runs atuais.
    split_path = Path(output_dirs["config_dir"]) / "split_ids.json"
    if split_path.is_file():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        split_ids = payload.get("splits", {}).get(split_name)
        if split_ids is not None:
            return [int(image_id) for image_id in split_ids]

    # Recorre aos metadados legados caso split_ids.json esteja ausente.
    numeric_dir = Path(output_dirs["numeric_dir"])
    for metadata_path in metadata_candidates(numeric_dir, split_name):
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        image_ids = metadata.get("execution_info", {}).get("image_ids")
        if image_ids is not None:
            return [int(image_id) for image_id in image_ids]
    raise FileNotFoundError(
        "Não foi possível validar o merge-only: split_ids.json e image_ids "
        "legados não foram encontrados no metadata."
    )


def save_split_metadata(
    split_name,
    output_dir,
    config,
    results,
    batch_timings,
    expected_batches,
    resolution=None,
):
    """Salva identidade, configuração efetiva e resumo dos resultados do split em JSON.

    Args:
        split_name (str): Nome do split executado (``train``, ``val``, ``test`` ou
            ``full``).
        output_dir (Path | str): Diretório numérico onde os arquivos de metadata
            são gravados.
        config (dict[str, Any]): Configuração efetiva usada na execução.
        results (pd.DataFrame): DataFrame consolidado com todos os registros do
            split.
        batch_timings (list[dict[str, Any]]): Lista de registros de duração dos
            lotes carregados dos manifests.
        expected_batches (list[int]): Lista de números dos lotes esperados na
            consolidação.
        resolution (str | None): Resolução da execução (``"mid"``, ``"high"`` ou
            ``None``).

    Returns:
        str: Caminho do arquivo de metadata principal gerado.
    """
    metadata_path = save_metadata(
        split_name,
        output_dir,
        config,
        resolution=resolution,
        results=results,
        batch_timings=batch_timings,
        expected_batches=expected_batches,
    )
    logger.info("Metadados salvos em: %s", metadata_path)
    return metadata_path


def _record_incomplete_integrity(output_dir, split_name, error):
    """Remove artefatos finais inválidos e persiste o diagnóstico de integridade.

    Args:
        output_dir (Path | str): Diretório numérico onde os artefatos finais são
            gravados.
        split_name (str): Nome do split cuja validação de integridade falhou.
        error (ResultIntegrityError): Exceção contendo o relatório detalhado de
            divergências de IDs.
    """
    output_dir = Path(output_dir)
    for stale_name in (
        f"summary_{split_name}.csv",
        metadata_filename(split_name),
        f"ostios_{split_name}_summary.csv",
        f"ostios_{split_name}_metadata.json",
    ):
        (output_dir / stale_name).unlink(missing_ok=True)
    marker = output_dir / integrity_filename(split_name)
    save_json_atomic(error.report, marker)


def _pipeline_error_count(results):
    """Conta exames com falha registrada no consolidado validado.

    Args:
        results (pd.DataFrame): DataFrame consolidado com os resultados por imagem.

    Returns:
        int: Quantidade de exames com erro não vazio registrado na coluna ``error``.
    """
    if "error" not in results.columns:
        return 0
    errors = results["error"].fillna("").astype(str).str.strip()
    return int(errors.ne("").sum())


def _finalize_split_results(
    split_name,
    image_ids,
    output_dir,
    config,
    resolution=None,
    *,
    missing_batches_exit=False,
    execution_time=None,
    batch_timing_summary=None,
    current_run_execution_time=None,
):
    """Consolida lotes, valida integridade e persiste metadata/resumo.

    Args:
        split_name (str): Nome do split consolidado (``train``, ``val``, ``test``
            ou ``full``).
        image_ids (list[int]): Lista esperada de IDs da coorte para validação de
            integridade.
        output_dir (Path | str): Diretório numérico contendo lotes e saídas finais.
        config (dict[str, Any]): Configuração efetiva da execução.
        resolution (str | None): Resolução do run (``"mid"``, ``"high"`` ou
            ``None``).
        missing_batches_exit (bool): Se ``True``, encerra via ``SystemExit(1)``
            quando nenhum lote existir.
        execution_time (float | None): Tempo total conhecido em segundos, quando
            já calculado.
        batch_timing_summary (dict[str, Any] | None): Resumo pré-computado dos
            tempos de lote.
        current_run_execution_time (float | None): Duração em segundos da
            invocação atual em retomadas.

    Returns:
        int: Número de exames com erro registrado no resultado consolidado.
    """
    # Etapa 1: mescla todos os CSVs parciais de lote no consolidado por imagem.
    output_path = merge_batch_results(split_name, output_dir)
    if output_path is None:
        if missing_batches_exit:
            print(f"❌ Nenhum lote encontrado para o split '{split_name}'")
            raise SystemExit(1)
        raise RuntimeError("Nenhum lote foi persistido para consolidação.")
    if not missing_batches_exit:
        logger.info("Resultados por imagem salvos em: %s", output_path)

    # Etapa 2: confere que nenhum ID da coorte foi duplicado, perdido ou acrescentado.
    df = pd.read_csv(output_path)
    try:
        validate_result_integrity(df, image_ids)
    except ResultIntegrityError as error:
        _record_incomplete_integrity(output_dir, split_name, error)
        raise
    (Path(output_dir) / integrity_filename(split_name)).unlink(missing_ok=True)

    # Etapa 3: reconstrói tempos totais dos lotes e grava metadata/resumo final do split.
    batch_timings = load_batch_timing_records(output_dir, split_name)
    expected_batches = [
        batch_result_number(path, split_name)
        for path in list_batch_result_files(split_name, output_dir)
    ]
    if batch_timing_summary is None:
        batch_timing_summary = summarize_batch_timing_records(batch_timings)
    if execution_time is None:
        execution_time = batch_timing_summary.get("total_known_duration_seconds")
    save_split_metadata(
        split_name,
        output_dir,
        config,
        df,
        batch_timings,
        expected_batches,
        resolution=resolution,
    )
    print_split_summary(
        df,
        split_name,
        config,
        execution_time,
        timing_summary=batch_timing_summary,
        current_run_execution_time=current_run_execution_time,
    )
    return _pipeline_error_count(df)


def run_merge_only_split(
    split_name,
    image_ids,
    output_dir,
    config,
    resolution=None,
):
    """Consolida lotes já existentes, valida integridade e recria o metadata compacto.

    Args:
        split_name (str): Nome do split consolidado (``train``, ``val``, ``test``
            ou ``full``).
        image_ids (list[int]): Lista esperada de IDs da coorte para validação de
            integridade.
        output_dir (Path | str): Diretório numérico contendo os CSVs parciais de
            lote.
        config (dict[str, Any]): Configuração efetiva registrada no metadata
            recalculado.
        resolution (str | None): Resolução do run (``"mid"``, ``"high"`` ou
            ``None``).

    Returns:
        int: Número de exames com falha registrada no CSV consolidado.
    """
    return _finalize_split_results(
        split_name,
        image_ids,
        output_dir,
        config,
        resolution=resolution,
        missing_batches_exit=True,
    )


def run_processing_split(
    split_name,
    ids,
    output_dir,
    config,
    args,
    base_path,
    visual_dir=None,
):
    """Executa o processamento em lotes de um split e consolida CSV final + metadata.

    Args:
        split_name (str): Nome do split a processar (``train``, ``val``, ``test``
            ou ``full``).
        ids (list[int]): Lista ordenada de IDs de exames ImageCAS pertencentes ao
            split.
        output_dir (Path | str): Diretório numérico de destino dos resultados e
            manifests.
        config (dict[str, Any]): Configuração efetiva normalizada e escalada para a
            resolução.
        args (argparse.Namespace): Namespace CLI com ``resume_batch`` e
            ``resolution``.
        base_path (Path): Diretório raiz do dataset ImageCAS.
        visual_dir (Path | str | None): Diretório opcional onde visualizações 3D
            HTML são salvas.

    Returns:
        int: Número de exames com erro registrado no consolidado final.
    """
    # Executa os lotes pendentes delegando a orquestração por exame.
    summary = run_pipeline(
        ids,
        split_name,
        config,
        base_path,
        output_dir,
        resume_from_batch=args.resume_batch,
        visual_output_dir=(
            Path(visual_dir)
            if config.get("SAVE_SEGMENTATION_VISUALS") and visual_dir is not None
            else None
        ),
        artery_mask_export=(
            ArteryMaskExport(
                Path(output_dir).parent, "ImageCAS", split_name, args.resolution
            )
            if config.get("SAVE_ARTERY_MASKS")
            else None
        ),
    )
    # Prioriza o acumulado de tempos dos lotes para refletir o tempo real em retomadas.
    current_run_execution_time = summary.get("execution_time")
    batch_timing_summary = summary.get("batch_timing_summary") or {}
    execution_time = (
        batch_timing_summary.get("total_known_duration_seconds")
        or current_run_execution_time
    )

    logger.info("Finalizando processamento em lotes...")
    return _finalize_split_results(
        split_name,
        ids,
        output_dir,
        config,
        resolution=args.resolution,
        missing_batches_exit=False,
        execution_time=execution_time,
        batch_timing_summary=batch_timing_summary,
        current_run_execution_time=current_run_execution_time,
    )


def run_requested_split(
    args,
    split_name,
    image_ids,
    output_dir,
    config,
    base_path,
    visual_dir=None,
):
    """Despacha a execução completa ou a consolidação (``--merge-only``) da coorte solicitada.

    Args:
        args (argparse.Namespace): Namespace CLI validado (define ``merge_only`` e
            parâmetros do run).
        split_name (str): Nome do split alvo (``train``, ``val``, ``test`` ou
            ``full``).
        image_ids (list[int]): Lista de IDs da coorte selecionada.
        output_dir (Path | str): Diretório numérico do run.
        config (dict[str, Any]): Configuração efetiva normalizada e escalada.
        base_path (Path): Diretório raiz do dataset ImageCAS.
        visual_dir (Path | str | None): Diretório opcional para salvar visualizações
            3D HTML.

    Returns:
        int: Número de exames com erro no resultado consolidado do split.
    """
    # Sinaliza no console o modo de operação e o conjunto selecionado.
    print(f"\n{'=' * 60}")
    action_label = "Consolidando" if args.merge_only else "Processando"
    print(f"🔬 {action_label} conjunto: {split_name.upper()}")
    print(f"{'=' * 60}")

    # Despacha para consolidação exclusiva de lotes ou processamento completo.
    if args.merge_only:
        return run_merge_only_split(
            split_name,
            image_ids,
            output_dir,
            config,
            resolution=args.resolution,
        )
    else:
        return run_processing_split(
            split_name,
            image_ids,
            output_dir,
            config,
            args,
            base_path,
            visual_dir,
        )


def _run_from_args(args):
    """Executa os argumentos validados e retorna o estado persistido.

    Args:
        args (argparse.Namespace): Namespace CLI validado do pipeline ImageCAS.

    Returns:
        tuple[Path, str, int]: Tupla ``(run_dir, split_name, error_count)`` com o
            diretório do run, split executado e quantidade de exames com erro.
    """
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Logging verbose habilitado (DEBUG)")

    # Merge-only trabalha apenas com lotes, snapshots e metadata já persistidos.
    base_path = args.base_path if args.merge_only else resolve_base_path(args.base_path)
    output_root_dir = args.output_dir
    effective_config = build_effective_config(args)

    # Prepara diretórios, logs e coorte antes de gravar snapshots de reprodutibilidade.
    print_run_settings(args, effective_config, base_path)
    output_dirs = resolve_output_dir(args, output_root_dir)
    args.notification_run_dir = Path(output_dirs["run_dir"])
    setup_file_logging(output_dirs["logs_dir"])
    split_name, image_ids = build_split_to_run(args, base_path)
    if args.merge_only:
        image_ids = load_merge_only_image_ids(output_dirs, split_name)
    # Grava os snapshots de config e split apenas na criação de um run novo.
    if not (args.resume_requested or args.merge_only):
        save_run_snapshots(
            output_dirs,
            effective_config,
            split_name,
            image_ids,
            split_config_path=args.split_config,
        )
    error_count = run_requested_split(
        args,
        split_name,
        image_ids,
        output_dirs["numeric_dir"],
        effective_config,
        base_path,
        output_dirs["visual_dir"],
    )
    return Path(output_dirs["run_dir"]), split_name, int(error_count or 0)


def main():
    """Executa o pipeline e avisa sobre o resultado quando solicitado."""
    # Interpreta a linha de comando e executa o fluxo com notificação de exceções.
    args = parse_pipeline_args(BASE_PATH, OUTPUT_DIR)
    try:
        run_dir, split_name, error_count = _run_from_args(args)
    except BaseException as error:
        if args.notify:
            fallback_dir = (
                Path(args.resume_dir) if args.resume_dir is not None else None
            )
            notify_run_exception(
                error,
                pipeline="ImageCAS",
                split=args.split,
                resolution=args.resolution,
                run_dir=getattr(args, "notification_run_dir", fallback_dir),
                notifier=notify_run_completion,
            )
        raise

    # Exibe o encerramento no console e dispara aviso final quando --notify está ativo.
    print(f"\n{'=' * 60}")
    print("✨ Processamento concluído!")
    print(f"{'=' * 60}\n")
    if args.notify:
        notify_run_completion(
            pipeline="ImageCAS",
            split=split_name,
            resolution=args.resolution,
            run_dir=run_dir,
            status="complete_with_errors" if error_count else "complete",
            details=(
                f"{error_count} exame(s) com erro de pipeline." if error_count else None
            ),
        )


if __name__ == "__main__":
    main()
