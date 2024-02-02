# coding=utf-8
# Copyright 2021 The Circuit Training Team Authors.
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
"""Library and util functions for reverb server."""
import os

import reverb
import tensorflow as tf
from absl import logging
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.policies import py_tf_eager_policy
from tf_agents.specs import tensor_spec
from tf_agents.train import learner
from tf_agents.train.utils import train_utils
from tf_agents.utils import common


def start_reverb_server(
    root_dir: str,
    replay_buffer_capacity: int,
    port: int,
    num_netlists: int = 1,
    buffer_min_size: int = 1,
    sample_limit: int = 1,
):
  """todo."""
