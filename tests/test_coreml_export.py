import sys
import unittest

import torch

from coreml.metric3d_coreml_export import Metric3DCoreMLExportModel

try:
    import coremltools as ct
except ImportError:  # pragma: no cover - optional dependency
    ct = None
try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


class DummyMetaArch(torch.nn.Module):
    def inference(self, inputs):
        image = inputs["input"]
        pred_depth = image.mean(dim=1, keepdim=True)
        return pred_depth, None, {}


class TestCoreMLExport(unittest.TestCase):
    def test_focal_length_scaling(self):
        torch.manual_seed(0)
        model = Metric3DCoreMLExportModel(DummyMetaArch(), canonical_focal_length=1000.0)
        image = torch.randn(2, 3, 4, 5)
        focal_length = torch.tensor([500.0, 1500.0])

        output = model(image, focal_length)
        expected = image.mean(dim=1, keepdim=True) * (
            focal_length.view(-1, 1, 1, 1) / 1000.0
        )
        self.assertTrue(torch.allclose(output, expected, rtol=1e-5, atol=1e-6))

    @unittest.skipIf(ct is None, "coremltools not installed")
    @unittest.skipIf(np is None, "numpy not installed")
    @unittest.skipIf(sys.platform != "darwin", "CoreML runtime requires macOS")
    def test_coreml_conversion_parity(self):
        torch.manual_seed(0)
        model = Metric3DCoreMLExportModel(DummyMetaArch(), canonical_focal_length=1000.0)
        image = torch.randn(1, 3, 4, 6)
        focal_length = torch.tensor([1200.0])

        traced = torch.jit.trace(model, (image, focal_length))
        convert_kwargs = {
            "inputs": [
                ct.TensorType(name="image", shape=image.shape),
                ct.TensorType(name="focal_length", shape=focal_length.shape),
            ],
            "outputs": [ct.TensorType(name="pred_depth")],
            "convert_to": "mlprogram",
        }
        if hasattr(ct, "precision"):
            convert_kwargs["compute_precision"] = ct.precision.FLOAT32

        mlmodel = ct.convert(traced, **convert_kwargs)
        torch_output = model(image, focal_length).detach().numpy()
        coreml_output = mlmodel.predict(
            {"image": image.numpy(), "focal_length": focal_length.numpy()}
        )["pred_depth"]

        np.testing.assert_allclose(coreml_output, torch_output, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
