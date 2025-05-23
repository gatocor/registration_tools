from dask.distributed.diagnostics.progress import format_time
from dask.distributed.diagnostics.progressbar import TextProgressBar# as DistributedProgressBar
from contextlib import suppress
import time
# dask.config.set(scheduler='synchronous')
import sys
from dask.callbacks import Callback
from dask.distributed import Client, Lock, LocalCluster
from tqdm import tqdm

class ProgressBar(Callback):
    def __init__(self, total_tasks):
        self.total_tasks = total_tasks  # Pass the total number of blocks as a parameter

    def _start_state(self, dsk, state):
        # Initialize tqdm with the number of blocks (tasks) to be processed
        self._tqdm = tqdm(total=self.total_tasks, desc="Dask Progress")

    def _posttask(self, key, result, dsk, state, worker_id):
        # Update the progress bar after each task completes
        self._tqdm.update(1)

    def _finish(self, dsk, state, errored):
        # Close the progress bar when the computation finishes
        self._tqdm.close()

class TqdmProgressBar(TextProgressBar):

    def __init__(
        self,
        keys,
        scheduler=None,
        interval="1s",
        width=None,
        loop=None,
        complete=True,
        start=True,
        total=100,
        **kwargs,
    ):
        self.max = total
        self.start = time.time()
        if width is None:
            width = min(1000,max(total,100))
        super().__init__(keys, scheduler, interval, width, loop, complete, start, **kwargs)

    def _draw_bar(self, remaining, all, **kwargs):
        frac = (1 - remaining / all) if all else 1.0
        bar = "#" * int(self.width * frac)
        n_completed = int(frac*self.max)
        t_elapsed = (time.time() - self.start)
        ratio = n_completed / t_elapsed
        if ratio >= 0:
            if ratio == 0:
                t_finish = 0
            else:
                t_finish = (self.max - n_completed) / ratio
            ratio = f"{ratio} it/s"
        else:
            t_finish = (self.max - n_completed) / ratio
            ratio = f"{1/ratio} s/it"
        msg = "\r[{0:<{1}}] | {2}/{3} | {4} | {5} < {6}".format(
            bar, self.width, int(frac*self.max), self.max, ratio, format_time(t_elapsed), format_time(t_finish)
        )
        with suppress(ValueError):
            sys.stdout.write(msg)
            sys.stdout.flush()

    def _draw_stop(self, **kwargs):
        print()
        return
