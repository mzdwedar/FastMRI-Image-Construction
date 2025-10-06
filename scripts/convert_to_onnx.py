import torch
from src.model import UnetModel


def convert_to_onnx(ckpoint_path:str):
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UnetModel.load_from_checkpoint(ckpoint_path)
    model.eval()
    model.to(device)

    input_sample = torch.randn(1, 1, 320, 320, device=device)
    with torch.no_grad():
        _ = model(input_sample)


    model_cpu = model.model.to("cpu")  # SMP model inside Lightning wrapper
    input_cpu = input_sample.to("cpu")

    onnx_path = "unet_model.onnx"
    torch.onnx.export(
        model_cpu,
        input_cpu,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=True,
        verbose=True
    )

    print(f"ONNX model exported successfully to {onnx_path}")

if __name__ == "__main__":
    checkpoint_path = "./mlflow_artifacts/checkpoints/UnetModel-ep00-v8.ckpt"
    convert_to_onnx(checkpoint_path)
