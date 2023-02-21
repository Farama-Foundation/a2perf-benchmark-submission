import os
import time

import gin
import numpy as np

from CoDE import test_websites
from CoDE import utils
from CoDE import vocabulary_node
from CoDE import web_environment
from CoDE import web_primitives
from CoDE import q_networks


def train():
    gin.parse_config_files_and_bindings(["/web_nav/gwob/configs/envdesign.gin"], None)
    time.sleep(10)
    os.system('xeyes')
    # Create an empty environment.
    env = web_environment.GMiniWoBWebEnvironment(
            base_url="file:///web_nav/gwob/",
            global_vocabulary=vocabulary_node.LockedVocabulary())
    time.sleep(10)

    # Create a q network.
    q_net = q_networks.DQNWebLSTM(vocab_size=env.local_vocab.max_vocabulary_size, return_state_value=True)
    time.sleep(10)
    # Sample a new design of the form {'number_of_pages': Integer, 'action': List[Integer], 'action_page': List[
    # Integer]}.
    # `action` denotes primitive indices and `action_page` denotes their corresponding page indices.
    # Each item in the `action_page` should be less than `number_of_pages`.
    # For this tutorial, we will randomly sample a design.
    number_of_pages = np.random.randint(4) + 1
    design = {'number_of_pages': number_of_pages,
              'action': np.random.choice(np.arange(len(web_primitives.CONCEPTS)), 5),
              'action_page': np.random.choice(np.arange(number_of_pages), 5)}
    time.sleep(10)
    # Design the actual environment.
    env.design_environment(
            design, auto_num_pages=True)
    time.sleep(10)
    # Reset the environment.
    state = env.reset()

    # Add batch dimension.
    state = {key: np.expand_dims(tensor, axis=0) for key, tensor in state.items()}

    # Get flattened logits and values.
    logits, values = q_net(state)

    # Get greedy action.
    action = np.argmax(logits)

    # Run the action.
    new_state, reward, done, info = env.step(action)
