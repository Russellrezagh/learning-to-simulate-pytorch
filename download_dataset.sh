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

# This script creates the directory structure for the .npz datasets.

# Make the root of this repository the current directory.
cd "$(dirname "$0")"

# This determines where data will be stored.
DATA_ROOT="${HOME}/learning_to_simulate/data"

# Experiments to create directories for.
EXPERIMENT_NAMES=(
    "WaterDrop"
    "RigidFall"
    "GoopFall"
    "SandFall"
    "Water"
    "Sand"
    "Goop"
)

# Create the directory structure
for EXPERIMENT_NAME in "${EXPERIMENT_NAMES[@]}"
do
  mkdir -p "${DATA_ROOT}/${EXPERIMENT_NAME}/train"
  mkdir -p "${DATA_ROOT}/${EXPERIMENT_NAME}/eval"
done

echo "Directory structure created in ${DATA_ROOT}"
