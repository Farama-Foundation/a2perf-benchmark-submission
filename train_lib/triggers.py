import json
import os

import tensorflow as tf
from absl import logging
from tf_agents.train import interval_trigger


class VocabularySaveTrigger(interval_trigger.IntervalTrigger):
  """Triggers saving the vocabulary dictionary to a specified directory at regular intervals."""

  def __init__(
      self,
      saved_vocab_dir: str,
      vocabulary: object,
      train_step: tf.Variable,
      interval: int,
  ):
    """Initializes a VocabularySaveTrigger."""
    self.saved_vocab_dir = saved_vocab_dir
    self.vocabulary = vocabulary
    self.train_step = train_step

    if not os.path.exists(self.saved_vocab_dir):
      os.makedirs(self.saved_vocab_dir)

    super(VocabularySaveTrigger, self).__init__(
        interval=interval, fn=self._save_fn,
    )

  def _save_fn(self) -> None:
    """Saves the vocabulary dictionary with padded train step number."""
    file_path = os.path.join(self.saved_vocab_dir,
                             f"vocab_{self.train_step.numpy():09d}.json")
    vocab_dict_proxy = self.vocabulary._global_vocab_node.local_vocab
    temp_dict = {k: v for k, v in vocab_dict_proxy.items()}
    with tf.io.gfile.GFile(file_path, 'w') as f:
      json.dump(temp_dict, f)

    logging.info('Vocabulary saved at step %d to %s', self.train_step.numpy(),
                 file_path)
