import tritonclient.http as httpclient
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.data import FastMRICustomDataset, FastMRITransform


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main predict function to run inference using Triton client"""
    
    data = FastMRICustomDataset(cfg.triton.image_path, FastMRITransform(mask_func=None))  # To check if the image path is valid
    dataloader = DataLoader(
                data,
                batch_size=16,
                shuffle=False,
                num_workers=10,
                persistent_workers=False
            )
    # Define the Client connection
    client = httpclient.InferenceServerClient(url=cfg.triton.server_url)

    for batch in dataloader:
        
        if len(batch) == 4:
            images, targets, fnames, slice_nums = batch
        else:
            images, fnames, slice_nums = batch

        images = images.detach().cpu()

        images_np = images.numpy().astype(np.float32)
        
        input_shape = images_np.shape
        infer_input = httpclient.InferInput(cfg.triton.input.name, input_shape, "FP32")
        infer_input.set_data_from_numpy(images_np)

        res = client.infer(cfg.triton.model_name, model_version=cfg.triton.model_version, inputs=[infer_input])

        output = res.as_numpy(cfg.triton.model_output_name)
        output = np.squeeze(output)


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter