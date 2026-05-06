"""
Export the torch hub model to CoreML format with focal length scaling.
Normalization is done in the model.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import coremltools as ct
except ImportError:  # pragma: no cover - optional dependency
    ct = None

try:
    from fire import Fire
except ImportError:  # pragma: no cover - optional dependency
    Fire = None


class Metric3DCoreMLExportModel(torch.nn.Module):
    """
    The model for exporting to CoreML format. Includes normalization and focal scaling.
    """

    def __init__(self, meta_arch: torch.nn.Module, canonical_focal_length: float = 1000.0):
        super().__init__()
        self.meta_arch = meta_arch
        self.register_buffer(
            "rgb_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "rgb_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
        )
        self.canonical_focal_length = float(canonical_focal_length)

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.rgb_mean) / self.rgb_std

    def _reshape_scale(
        self, scale: torch.Tensor, pred_depth: torch.Tensor
    ) -> torch.Tensor:
        if scale.dim() == 0:
            scale = scale.view(1)
        if scale.dim() == 1:
            if pred_depth.dim() == 4:
                return scale.view(-1, 1, 1, 1)
            return scale.view(-1, 1, 1)
        if scale.dim() == 2 and scale.shape[1] == 1:
            if pred_depth.dim() == 4:
                return scale.view(-1, 1, 1, 1)
            return scale.view(-1, 1, 1)
        if pred_depth.dim() == 4 and scale.dim() == 3:
            return scale.unsqueeze(1)
        return scale

    def forward(self, image: torch.Tensor, focal_length: torch.Tensor) -> torch.Tensor:
        image = self._normalize_image(image)
        with torch.no_grad():
            pred_depth, _, _ = self.meta_arch.inference({"input": image})
        scale = focal_length / self.canonical_focal_length
        scale = scale.to(dtype=pred_depth.dtype)
        scale = self._reshape_scale(scale, pred_depth)
        return pred_depth * scale


def update_vit_sampling(model: torch.nn.Module) -> torch.nn.Module:
    """
    For ViT models running on some TensorRT version, we need to change
    the interpolation method from bicubic to bilinear.
    """
    import math
    import torch.nn as nn

    def interpolate_pos_encoding_bilinear(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=(sx, sy),
            mode="bilinear",
            antialias=self.interpolate_antialias,
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    model.depth_model.encoder.interpolate_pos_encoding = (
        interpolate_pos_encoding_bilinear.__get__(
            model.depth_model.encoder, model.depth_model.encoder.__class__
        )
    )
    return model


def _resolve_input_size(
    model_name: str,
    input_height: Optional[int],
    input_width: Optional[int],
) -> Tuple[int, int]:
    if input_height is None and input_width is None:
        if "vit" in model_name:
            return 616, 1064
        return 544, 1216
    if input_height is None or input_width is None:
        raise ValueError("Both input_height and input_width must be provided.")
    return input_height, input_width


def export_coreml(
    model_name: str = "metric3d_vit_small",
    modify_upsample: bool = False,
    input_height: Optional[int] = None,
    input_width: Optional[int] = None,
    canonical_focal_length: float = 1000.0,
    output_path: Optional[str] = None,
):
    if ct is None:
        raise ImportError(
            "coremltools is required for CoreML export. "
            "Install it with `pip install coremltools`."
        )

    model = torch.hub.load("yvanyin/metric3d", model_name, pretrain=True)
    model.eval()

    if modify_upsample and "vit" in model_name:
        model = update_vit_sampling(model)

    input_height, input_width = _resolve_input_size(
        model_name, input_height, input_width
    )
    dummy_image = torch.zeros((1, 3, input_height, input_width), dtype=torch.float32)
    dummy_focal = torch.tensor([canonical_focal_length], dtype=torch.float32)

    export_model = Metric3DCoreMLExportModel(
        model, canonical_focal_length=canonical_focal_length
    )
    export_model.eval()

    traced = torch.jit.trace(export_model, (dummy_image, dummy_focal))

    convert_kwargs = {
        "inputs": [
            ct.TensorType(name="image", shape=dummy_image.shape),
            ct.TensorType(name="focal_length", shape=dummy_focal.shape),
        ],
        "outputs": [ct.TensorType(name="pred_depth")],
        "convert_to": "mlprogram",
    }
    if hasattr(ct, "precision"):
        convert_kwargs["compute_precision"] = ct.precision.FLOAT32

    mlmodel = ct.convert(traced, **convert_kwargs)
    output_path = output_path or f"{model_name}.mlpackage"
    mlmodel.save(output_path)
    return output_path


def main(
    model_name: str = "metric3d_vit_small",
    modify_upsample: bool = False,
    input_height: Optional[int] = None,
    input_width: Optional[int] = None,
    canonical_focal_length: float = 1000.0,
    output_path: Optional[str] = None,
):
    return export_coreml(
        model_name=model_name,
        modify_upsample=modify_upsample,
        input_height=input_height,
        input_width=input_width,
        canonical_focal_length=canonical_focal_length,
        output_path=output_path,
    )


if __name__ == "__main__":
    if Fire is None:
        raise ImportError("fire is required to run this script. Install it with `pip install fire`.")
    Fire(main)
