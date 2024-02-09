import multiprocessing
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
      f'Attempting to start vocabulary manager server at {_VOCABULARY_SERVER_ADDRESS.value}:{_VOCABULARY_SERVER_PORT.value}.' * 10000)

  manager = VocabularyManager(
      address=(
          _VOCABULARY_SERVER_ADDRESS.value,
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

  manager.start()
  logging.info(
      f'Started vocabulary manager server at {_VOCABULARY_SERVER_ADDRESS.value}:{_VOCABULARY_SERVER_PORT.value}.')

  # Keep the main script running
  while True:
    time.sleep(60)
    logging.info('Vocabulary manager server is running.')


if __name__ == '__main__':
  flags.mark_flags_as_required(
      ['vocabulary_manager_auth_key',
       'max_vocab_size',
       'vocabulary_server_address',
       'vocabulary_server_port'
       ])
  app.run(main)
