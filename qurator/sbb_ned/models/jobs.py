from multiprocessing import Semaphore
import numpy as np


class Job:

    def __init__(self, queue, job_id, priority, task_info, task_len, params=None):

        self._id = job_id
        self.priority = priority
        self._sem = Semaphore(1)
        self._results = []

        self._task_info = task_info
        self._task_len = task_len
        self._num_pending = task_len

        self._queue = queue
        self._params = (params,) if params is not None else None

    def num_pending(self):

        with self._sem:
            return self._num_pending

    def add_result(self, result):

        with self._sem:
            self._results.append(result)

    def get(self):

        with self._sem:

            if type(self._task_info) == list:
                ret = self._task_info.pop(0) if len(self._task_info) > 0 else None
            else:
                ret = next(self._task_info, None) if self._task_len > 0 else None

            self._num_pending = self._num_pending - 1 if self._num_pending > 0 else 0

        if ret is None or self._params is None:
            return ret
        else:
            return (*ret, *self._params)

    def put(self, task_info):
        with self._sem:
            if type(self._task_info) == list:
                self._task_info.append(task_info)
                self._task_len += 1
                self._num_pending += 1
            else:
                raise RuntimeError('Generator job does not support put!')

    def sequence(self):

        while self._task_len > 0:

            if JobQueue.quit:
                return

            self._queue.do_next_task()

            with self._sem:
                results = self._results
                self._results = []
                self._task_len -= len(results)

            for result in results:

                yield (self._id, *result)

    def remove(self):

        self._queue.remove_job(self._id, self.priority)


class JobQueue:

    quit = False

    def __init__(self, result_sequence=None, min_level=2, name="JobQueue", verbose=False, feeder_queue=None,
                 limit=None):

        self._verbose = verbose
        self._result_sequence = result_sequence
        self._next_call_sem = Semaphore(0) if result_sequence is not None else None
        self._execution_sem = Semaphore(1) if result_sequence is not None else None

        self._prio_levels = [l for l in range(0, min_level)]
        self._priorities = {l: list() for l in self._prio_levels}
        self._name = name

        self._main_sem = Semaphore(1)
        self._job_counter = 0
        self._process_queue = dict()
        self._process_queue_sem = Semaphore(0)

        self._feeder_queue = feeder_queue
        self._limit_sem = [Semaphore(limit) for _ in range(0, min_level)] if limit is not None else None

    def has(self, job_id):

        return job_id in self._process_queue

    def max_prio(self):

        with self._main_sem:
            for prio in self._prio_levels:

                for job_id in self._priorities[prio]:

                    if self._process_queue[job_id].num_pending() > 0:
                        return prio
            return 0

    def add_job(self, task_info, priority, task_len=None, params=None):

        task_len = task_len if task_len is not None else len(task_info)

        with self._main_sem:

            job = Job(self, self._job_counter, priority, task_info, task_len, params=params)

            self._process_queue[self._job_counter] = job
            self._priorities[priority].append(self._job_counter)
            self._job_counter += 1

            for _ in range(0, task_len):
                self._process_queue_sem.release()

                if self._result_sequence is not None:
                    self._next_call_sem.release()

        return job

    def prio_above_pending(self, prio):

        if self._feeder_queue is not None and self._feeder_queue.prio_above_pending(prio):
            print('Above pending prio: {}'.format(prio))
            return True
        else:
            if not self._process_queue_sem.acquire(block=False):
                return False  # nothing to process ...
            else:
                self._process_queue_sem.release()

        # if self._limit_sem is not None:
        #     if self._limit_sem.acquire(block=False):
        #         self._limit_sem.release()
        #     else:
        #         return False  # nothing to process ...

        with self._main_sem:
            prio -= 1
            while prio >= 0:

                for job_id in self._priorities[prio]:

                    if self._process_queue[job_id].num_pending() > 0:
                        return True
                prio -= 1

        return False

    def add_to_job(self, job_id, task_info):

        with self._main_sem:
            if self.has(job_id):

                job = self._process_queue[job_id]

                job.put(task_info)
                self._process_queue_sem.release()

                if self._result_sequence is not None:
                    self._next_call_sem.release()

    def remove_job(self, job_id, priority):

        with self._main_sem:

            if job_id in self._process_queue:
                self._process_queue.pop(job_id)
                self._priorities[priority].remove(job_id)
            else:
                print('Warning: attempt to remove non-existent job!!!')

    def do_next_task(self):

        if self._result_sequence is None:
            raise RuntimeError('JobQueue does not have result sequence!')

        if self._next_call_sem.acquire(timeout=10):

            with self._execution_sem:
                job_id, result = next(self._result_sequence)

            self._process_queue[job_id].add_result(result)

            if self._limit_sem is not None:
                self._limit_sem[self._process_queue[job_id].priority].release()

    def get_next_task(self):

        def _gn():
            for _prio in self._prio_levels:

                order = np.random.permutation(len(self._priorities[_prio]))

                for pos in order:

                    _job_id = self._priorities[_prio][pos]

                    _task_info = self._process_queue[_job_id].get()

                    if _task_info is not None:

                        if self._verbose:
                            print("{}: job_id: {} #prio {} jobs: {}".
                                  format(self._name, _job_id, _prio, len(self._priorities[_prio])))

                        return _job_id, _prio, _task_info

            return None, None, None

        if self._feeder_queue is not None:
            if self._feeder_queue.prio_above_pending(self.max_prio()):
                return None, None, JobQueue.quit
        else:
            if not self.wait(self._process_queue_sem):
                return None, None, JobQueue.quit

        with self._main_sem:
            job_id, prio, task_info = _gn()

        if self._limit_sem is not None and job_id is not None:
            if not self.wait(self._limit_sem[prio], "_limit_sem[{}]".format(prio)):
                return None, None, JobQueue.quit

        return job_id, task_info, JobQueue.quit

    @staticmethod
    def wait(sem=None, msg=None):

        while True:
            if sem is not None and sem.acquire(timeout=10):
                return True

            if msg is not None:
                print(msg)

            if JobQueue.quit:
                return False

            if sem is None:
                return True
