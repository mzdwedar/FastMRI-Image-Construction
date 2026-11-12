"""Export a trained checkpoint to ONNX for Triton and check it matches PyTorch.

    uv run python -m scripts.convert_to_onnx --checkpoint artifacts/model.ckpt

Writes model.onnx into every Triton model directory listed in --model-dirs, then runs
the same random batch through PyTorch and ONNX Runtime and fails if outputs diverge.

--random-init exports untrained weights and also writes a matching .ckpt. Serving
latency and throughput don't depend on the weights, so this is what the benchmark
uses when the FastMRI dataset (registration required) isn't available.
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from src.model import UnetModel

MODEL_REPO = Path("deploy/services/triton/models")
DEFAULT_MODEL_DIRS = [MODEL_REPO / "unet_model", MODEL_REPO / "unet_model_nobatch"]
INPUT_SHAPE = (1, 320, 320)


def random_init_checkpoint(path: Path) -> UnetModel:
    cfg = OmegaConf.load("configs/trainer.yaml").trainer.train
    model = UnetModel(cfg.optimizer.lr, cfg.optimizer.weight_decay, cfg.optimizer.name, 32, 4, None)
    trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False)
    trainer.strategy.connect(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    return model


def export(module: UnetModel, onnx_path: Path, opset: int = 17) -> None:
    net = module.model.eval().cpu()  # the smp.Unet inside the LightningModule
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        torch.randn(1, *INPUT_SHAPE),
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )


def check_parity(module: UnetModel, onnx_path: Path, batch: int = 4, atol: float = 1e-4) -> float:
    x = np.random.default_rng(0).random((batch, *INPUT_SHAPE), dtype=np.float32)
    with torch.inference_mode():
        expected = module.model.eval()(torch.from_numpy(x)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(["output"], {"input": x})[0]
    max_diff = float(np.abs(expected - actual).max())
    if max_diff > atol:
        raise AssertionError(f"ONNX output differs from PyTorch by {max_diff:.2e} (> {atol})")
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/model.ckpt"))
    parser.add_argument("--random-init", action="store_true", help="write an untrained checkpoint first")
    parser.add_argument("--model-dirs", type=Path, nargs="+", default=DEFAULT_MODEL_DIRS)
    args = parser.parse_args()

    if args.random_init:
        random_init_checkpoint(args.checkpoint)
        print(f"wrote untrained checkpoint {args.checkpoint}")
    module = UnetModel.load_from_checkpoint(args.checkpoint, map_location="cpu")

    first = args.model_dirs[0] / "1" / "model.onnx"
    export(module, first)
    print(f"exported {first}  (max |torch - onnxruntime| = {check_parity(module, first):.2e})")
    for d in args.model_dirs[1:]:
        target = d / "1" / "model.onnx"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(first.read_bytes())
        print(f"copied   {target}")


if __name__ == "__main__":
    main()
