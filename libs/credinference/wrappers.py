"""Wrappers used by pecanpy.
https://github.com/krishnanlab/PecanPy/blob/297f89d4721664c0e00ad1978a9c76d5e0eede32/src/pecanpy/wrappers.py"""

import time
import io
import os


class Timer:
    """Timer for logging runtime of function."""

    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose

    def __call__(self, func):
        """Call timer decorator."""

        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start

            hrs = int(duration // 3600)
            mins = int(duration % 3600 // 60)
            secs = duration % 60
            stat = f"Took {hrs:02d}:{mins:02d}:{secs:05.2f} to {self.name}"
            print(stat)
            log_dir = "timing"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(os.path.join(log_dir, "log.txt"), "a") as log:
                log.write(f"{stat}\n")
            return result

        return wrapper if self.verbose else func
