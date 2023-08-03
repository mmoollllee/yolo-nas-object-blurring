# Source: https://gist.github.com/jeremyephron/3dbf7c4c778ce38e63cea7c9ab1fd99d
import errno
import fcntl
from pathlib import Path

def lock_script() -> bool:
   """
   Locks a file pertaining to this script so that it cannot be run simultaneously.
   
   Since the lock is automatically released when this script ends, there is no 
   need for an unlock function for this use case.
   
   Returns:
      True if the lock was acquired, False otherwise.
   
   """
   
   global lockfile  # file must remain open until program quits
   lockfile = open(f'.lock', 'w')
   
   try:
      # Try to grab an exclusive lock on the file, raise error otherwise
      fcntl.lockf(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
      
   except OSError as e:
      if e.errno == errno.EACCES or e.errno == errno.EAGAIN:
         return False
      raise
      
   else:
      return True
