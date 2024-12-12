#!/bin/bash
# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script will run training for all experiments, then evaluate them all.
# It assumes that you have already prepared the data and put it in directories
# named `learning_to_simulate/data`.

# Stop the script if any statement returns an error.
set -e

# Make the root of this repository the current directory.
cd "$(dirname "$0")"

# This determines where data is located and where models will be saved to.
RESULTS_PATH="${HOME}/learning_to_simulate" # Or wherever you want to store results
DATA_PATH="${RESULTS_PATH}/data"
MODELS_PATH="${RESULTS_PATH}/models"

# Create the directories if they don't exist.
mkdir -p "${DATA_PATH}"
mkdir -p "${MODELS_PATH}"

# Experiment to run.
EXPERIMENT_NAME="Sand"

# Train.
echo "Training model on ${EXPERIMENT_NAME}"
MODEL_DIR="${MODELS_PATH}/${EXPERIMENT_NAME}"
python train.py \
  --mode=train \
  --data_path="${DATA_PATH}/${EXPERIMENT_NAME}" \
  --model_dir="${MODEL_DIR}" \
  --num_steps=200 \
  --batch_size=2 \
  --config=model_config.json

# Evaluate.
echo "Evaluating model on ${EXPERIMENT_NAME}"
python train.py \
  --mode=eval \
  --data_path="${DATA_PATH}/${EXPERIMENT_NAME}" \
  --model_dir="${MODEL_DIR}" \
  --batch_size=2 \
  --config=model_config.json
