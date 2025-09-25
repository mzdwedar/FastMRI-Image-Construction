#!/bin/bash

# update these variables
export run_id=6ae3660afa804c0289fbdffd9f86107e 
export experiment_id=666907166598064635
export checkpoint_path=mlruns/666907166598064635/6ae3660afa804c0289fbdffd9f86107e/artifacts/checkpoints/UnetModel-ep00-v6.ckpt

docker compose -f ../../docker-compose.yml up --build