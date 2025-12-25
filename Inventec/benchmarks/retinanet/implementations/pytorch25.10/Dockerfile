# Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:25.10-py3

FROM ${FROM_IMAGE_NAME} AS base-image

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

RUN git config --global user.name "a" && \
    git config --global user.email "a"

ENV PIP_CONSTRAINT=""

################################################################################
# DALI box iou builder image
################################################################################
FROM base-image AS dali-box-iou-builder
WORKDIR /workspace/csrc
COPY csrc/dali_box_iou ./dali_box_iou

# Build CUDA box iou (written as a DALI extension)
RUN cd dali_box_iou && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make
################################################################################


################################################################################
# DALI proposal matcher builder image
################################################################################
FROM base-image AS dali-proposal-matcher-builder
WORKDIR /workspace/csrc
COPY csrc/dali_proposal_matcher ./dali_proposal_matcher

# Build CUDA proposal matcher (written as a DALI extension)
RUN cd dali_proposal_matcher && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make
################################################################################


################################################################################
# RetinaNet
################################################################################
FROM base-image

# Build args
ARG MAX_JOBS=4
ARG TORCH_CUDNN_V8_API_ENABLED=1

# Set env vars
ENV MAX_JOBS=${MAX_JOBS}
ENV TORCH_CUDNN_V8_API_ENABLED=${TORCH_CUDNN_V8_API_ENABLED}

# Apex: replace "SKIP" with "90055a9188f907e1fd78d9099aa45252efd62efb" (2025-11-05 snapshot)
ARG APEX_REVISION=90055a9188f907e1fd78d9099aa45252efd62efb
ENV CUSTOM_APEX_REVISION=${APEX_REVISION}

RUN if [ "${APEX_REVISION}" != SKIP ]; then \
      git clone https://github.com/NVIDIA/apex.git && \
      cd apex && \
      echo APEX_REVISION=${APEX_REVISION} && \
      git checkout ${APEX_REVISION} && \
      echo APEX_COMMIT_HASH=$(git rev-parse HEAD) && \
      APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_FOCAL_LOSS=1 APEX_FUSED_CONV_BIAS_RELU=1 pip install -v --no-build-isolation . \
    ; fi

# Install dependencies for system configuration logger
RUN apt-get update && \
    apt-get install -y --no-install-recommends infiniband-diags pciutils && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace/ssd

# Copy code
COPY . .

# Remove the container's pycocotools in favor of the nvidia optimized version (installed from requirements.txt)
RUN pip uninstall -y pycocotools

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy DALI box iou and proposal matcher
COPY --from=dali-box-iou-builder /workspace/csrc/dali_box_iou/build/lib_box_iou.so /usr/local/lib/lib_box_iou.so
COPY --from=dali-proposal-matcher-builder /workspace/csrc/dali_proposal_matcher/build/lib_proposal_matcher.so /usr/local/lib/lib_proposal_matcher.so
