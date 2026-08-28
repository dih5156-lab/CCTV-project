#!/usr/bin/env python3
"""Export a Rethinking_of_PAR pedestrian-attribute checkpoint to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch import nn


class AttributeLogitsWrapper(nn.Module):
    """Rethinking_of_PAR returns ([logits], feature); expose only logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _feature = self.model(images)
        if isinstance(logits, (list, tuple)):
            return logits[0]
        return logits


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def _build_model(repo_root: Path, nattr: int, backbone_name: str, classifier_name: str) -> nn.Module:
    sys.path.insert(0, str(repo_root))

    # Imports register backbones/classifiers in the external repository registry.
    from models.backbone import resnet  # noqa: F401
    from models.base_block import FeatClassifier
    from models.model_factory import build_backbone, build_classifier

    from models import base_block  # noqa: F401

    backbone, c_output = build_backbone(backbone_name, multi_scale=False)
    classifier = build_classifier(classifier_name)(
        nattr=nattr,
        c_in=c_output,
        bn=False,
        pool="avg",
        scale=1,
    )
    return FeatClassifier(backbone, classifier, bn_wd=True)


def export_onnx(
    *,
    repo_root: Path,
    checkpoint_path: Path,
    output_path: Path,
    nattr: int,
    height: int,
    width: int,
    backbone_name: str,
    classifier_name: str,
    use_ema: bool,
    opset: int,
) -> None:
    model = _build_model(repo_root, nattr, backbone_name, classifier_name)
    # Local training checkpoints include metadata objects, so disable
    # weights-only mode introduced as the default in PyTorch 2.6.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_key = "state_dict_ema" if use_ema and "state_dict_ema" in checkpoint else "state_dicts"
    state_dict = _strip_module_prefix(checkpoint[state_key])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    wrapped = AttributeLogitsWrapper(model).eval()
    dummy = torch.randn(1, 3, height, width, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        output_path,
        input_names=["images"],
        output_names=["scores"],
        dynamic_axes={"images": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"exported: {output_path}")
    print(f"checkpoint_epoch: {checkpoint.get('epoch')}")
    print(f"checkpoint_metric: {checkpoint.get('metric')}")
    print(f"state_key: {state_key}")
    print(f"input_shape: [batch, 3, {height}, {width}]")
    print(f"output_shape: [batch, {nattr}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("external/Rethinking_of_PAR"),
        help="Rethinking_of_PAR repository path",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nattr", type=int, default=26)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--classifier", default="linear")
    parser.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA")
    parser.add_argument("--opset", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_onnx(
        repo_root=args.repo_root.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        output_path=args.output.resolve(),
        nattr=args.nattr,
        height=args.height,
        width=args.width,
        backbone_name=args.backbone,
        classifier_name=args.classifier,
        use_ema=not args.no_ema,
        opset=args.opset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
