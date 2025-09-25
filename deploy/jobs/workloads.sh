#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD
mkdir results

# Test data
echo "Step0: Running (data) tests..."
export RESULTS_FILE=results/test_data_results.txt
pytest tests/data --verbose --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE

# Test code
echo "Step1: Running (code) tests..."
export stop_on_first_improvement=true
export RESULTS_FILE=results/test_code_results.txt
python -m pytest tests/code --verbose --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE

# Train
echo "Step2: Training model..."
export stop_on_first_improvement=false
export EXPERIMENT_NAME="fastmri"
export TRAIN_RESULTS_FILE=results/training_results.json
export max_epochs=10
export batch_size=16
python scripts/train.py

# # Get and save run ID
echo "Step3: getting run ID..."
eval $(python -c "
        import os
        from src import utils
        d = utils.load_dict(os.getenv('TRAIN_RESULTS_FILE'))
        print(f'export RUN_ID={d[\"run_id\"]}')
        print(f'export EXPERIMENT_ID={d[\"experiment_id\"]}')
        print(f'export checkpoint_path={d[\"checkpoint_path\"]}')
        ")
export RUN_ID_FILE=results/run_id.txt
echo $RUN_ID > $RUN_ID_FILE

# # Evaluate
echo "Step4: Evaluating model..."
export EVAL_RESULTS_FILE=results/evaluation_results.json
export eval_data_path="data/val"
export run_id=$RUN_ID
python scripts/evaluate.py

# TODO: Test model

# TODO: move model artifacts to production path
