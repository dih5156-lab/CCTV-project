"""CCTV 시스템 CLI 진입점."""

import logging

from src.bootstrap.cli import apply_args_to_config, build_parser, validate_args
from src.bootstrap.runtime import (
    build_single_source_camera,
    configure_runtime_environment,
    load_camera_list,
    register_shutdown_handlers,
    setup_logging,
    start_processor_runtime,
)
from src.config import AppConfig

logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60


def main() -> None:
    """메인 진입점."""
    configure_runtime_environment()
    setup_logging()

    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    cfg = apply_args_to_config(args, AppConfig.from_env())
    if not cfg.validate():
        logger.warning("설정 검증 실패. 일부 기능이 작동하지 않을 수 있습니다.")

    logger.info(SEPARATOR)
    logger.info("CCTV 헬멧 감지 시스템")
    logger.info(SEPARATOR)
    logger.info(cfg.summary())
    logger.info(SEPARATOR)

    if args.cameras:
        cameras = load_camera_list(args.cameras)
    else:
        cameras = build_single_source_camera(args.video)

    processor_refs: list = []
    register_shutdown_handlers(processor_refs)

    start_processor_runtime(
        cameras,
        cfg,
        cameras_json_path=args.cameras or "cameras.json",
        api_port=args.api_port,
        zone_presets_path=args.zone_presets,
        processor_refs=processor_refs,
    )


if __name__ == "__main__":
    main()
