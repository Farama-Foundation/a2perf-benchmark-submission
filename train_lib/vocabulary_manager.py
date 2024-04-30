import json
import multiprocessing
import os
import time
from multiprocessing.managers import BaseManager

from absl import app
from absl import flags
from absl import logging

_VOCABULARY_SERVER_PORT = flags.DEFINE_integer(
    'vocabulary_server_port', None, 'Port number for the vocabulary manager.'
)
_VOCABULARY_SERVER_ADDRESS = flags.DEFINE_string(
    'vocabulary_server_address', None, 'Address for the vocabulary manager.'
)
_VOCABULARY_MANAGER_AUTH_KEY = flags.DEFINE_string(
    'vocabulary_manager_auth_key', None,
    'Authentication key for the manager server.'
)
_MAX_VOCAB_SIZE = flags.DEFINE_integer(
    'max_vocab_size', None, 'Maximum vocabulary size.'
)
_ROOT_DIR = flags.DEFINE_string(
    'root_dir', '/tmp/xm_local', 'Base directory for logs and results.'
)


class VocabularyManager(BaseManager):
  pass


def LockProxy(lock):
  """Create a lock proxy that supports context management."""

  class LockProxy:

    def acquire(self):
      return lock.acquire()

    def release(self):
      return lock.release()

    def __enter__(self):
      self.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
      self.release()

  return LockProxy()


def main(_):
  logging.info(
      f'Attempting to start vocabulary manager server at '
      f'{_VOCABULARY_SERVER_ADDRESS.value}:{_VOCABULARY_SERVER_PORT.value}.'
  )
  logging.info(f'Using authkey: {_VOCABULARY_MANAGER_AUTH_KEY.value.encode()}')

  manager = VocabularyManager(
      address=(
          '0.0.0.0',
          # Allow connections from any address (firewall rules apply)
          _VOCABULARY_SERVER_PORT.value,
      ),
      authkey=_VOCABULARY_MANAGER_AUTH_KEY.value.encode()
  )

  general_manager = multiprocessing.Manager()
  _shared_dict = general_manager.dict()
  _shared_lock = multiprocessing.Lock()

  # Registering shared dictionary
  VocabularyManager.register(
      'get_shared_dict',
      callable=lambda: _shared_dict,
      exposed=(
          '__getitem__',
          '__setitem__',
          '__delitem__',
          '__len__',
          '__iter__',
          '__contains__',
          '__str__',
          '__repr__',
          'clear',
          'copy',
          'get',
          'items',
          'keys',
          'pop',
          'popitem',
          'setdefault',
          'update',
          'values',
      ),
  )
  # Registering shared lock with __enter__ and __exit__ methods exposed
  VocabularyManager.register(
      'get_shared_lock',
      callable=lambda: LockProxy(_shared_lock),
      exposed=('__enter__', '__exit__', 'acquire', 'release'),
  )

  # Before starting the vocab server, check for saved vocab file and load it
  # if it exists. get all files in the dir, sort them and load the latest checkpoint
  # into the shared dict.
  save_vocab_dir = os.path.join(_ROOT_DIR.value, 'vocabulary')
  if os.path.exists(save_vocab_dir):
    vocab_files = os.listdir(save_vocab_dir)
    if vocab_files:
      vocab_files.sort()
      latest_vocab_file = vocab_files[-1]
      with open(os.path.join(save_vocab_dir, latest_vocab_file), 'r') as f:
        _shared_dict.update(json.load(f))
      logging.info(
        f'Successfully re-loaded vocabulary from {latest_vocab_file}.')

  manager.start()
  logging.info(
      f'Started vocabulary manager server at {_VOCABULARY_SERVER_ADDRESS.value}:{_VOCABULARY_SERVER_PORT.value}.')

  # Keep the main script running
  while True:
    time.sleep(60)
    logging.info(
        f'Vocabulary manager server running at {_VOCABULARY_SERVER_ADDRESS.value}:{_VOCABULARY_SERVER_PORT.value}.')
    logging.info(f'\tCurrent vocabulary size: {len(_shared_dict)}')
    logging.info(
        f'\tConnect with authkey: {_VOCABULARY_MANAGER_AUTH_KEY.value.encode()}')


if __name__ == '__main__':
  flags.mark_flags_as_required(
      ['vocabulary_manager_auth_key',
       'max_vocab_size',
       'vocabulary_server_address',
       'vocabulary_server_port'
       ])
  app.run(main)
