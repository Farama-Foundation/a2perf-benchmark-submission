import multiprocessing
from multiprocessing.managers import BaseManager
import time

from absl import app
from absl import flags
from absl import logging

# Define flags
_PORT = flags.DEFINE_integer(
    'port', None, 'Port number for the manager server.'
)
_AUTH_KEY = flags.DEFINE_string(
    'auth_key', None, 'Authentication key for the manager server.'
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
  manager = VocabularyManager(
      address=('', _PORT.value), authkey=_AUTH_KEY.value.encode()
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
  logging.info(f'Started vocabulary manager server on port {_PORT.value}.')

  # Keep the main script running
  while True:
    time.sleep(10)
    logging.info('Vocabulary manager server is running.')


if __name__ == '__main__':
  flags.mark_flags_as_required(['port', 'auth_key', 'max_vocab_size'])
  app.run(main)
