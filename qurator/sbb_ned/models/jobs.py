from threading import Semaphore

import numpy as np
import inspect
import logging

logger = logging.getLogger(__name__)


class InfiniteLoop:
    def __init__(self, warn_limit, message):
        self.warn_limit = warn_limit
        self._message = message
        self._counter = 0

    def __call__(self, *args, **kwargs):

        self._counter += 1
        # print("Iteration {}: {}".format(self._counter, self._message))

        if self._counter == self.warn_limit:
            logger.warning("Loop Warning: {}".format(self._message))

        return True

    def warn(self):

        return self._counter > self.warn_limit

    def reset(self):
        self._counter = 0


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

        loop = InfiniteLoop(100, "{}:{}:{}".format(self._id,
                                                   inspect.getframeinfo(inspect.currentframe()).filename,
                                                   inspect.getframeinfo(inspect.currentframe()).lineno))

        while self._task_len > 0 and loop():

            if JobQueue.quit:
                return

            self._queue.do_next_task(loop=loop)

            with self._sem:
                results = self._results
                self._results = []
                self._task_len -= len(results)

            for result in results:

                loop.reset()

                yield (self._id, *result)

    def remove(self):

        self._queue.remove_job(self._id, self.priority)


class JobQueue:

    quit = False

    def __init__(self, result_sequence=None, min_level=2, name="JobQueue", verbose=False, feeder_queue=None,
                 limit=None):

        limit = max(1, limit) if limit is not None else None

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
            logger.debug('{}: Above pending prio: {}'.format(self._name, prio))
            return True
        else:
            if not self._process_queue_sem.acquire(blocking=False):
                return False  # nothing to process ...
            else:
                self._process_queue_sem.release()

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

    def remove_job(self, job_id, priority=None):

        with self._main_sem:

            if job_id in self._process_queue:
                job = self._process_queue.pop(job_id)

                if priority is not None:
                    self._priorities[priority].remove(job_id)
                else:
                    for priority, priority_ids in self._priorities.items():
                        if job_id in priority_ids:
                            priority_ids.remove(job_id)

                if job.num_pending() > 0:
                    logger.warning('Warning job_id: {} num_pending > 0 !!!'.format(job_id))
            else:
                 logger.warning('Warning: attempt to remove non-existent job!!!')

    def do_next_task(self, loop=None):

        if self._result_sequence is None:
            raise RuntimeError('JobQueue does not have result sequence!')

        if self._next_call_sem.acquire(timeout=1):

            with self._execution_sem:
                job_id, result = next(self._result_sequence)

            self._process_queue[job_id].add_result(result)

            if self._limit_sem is not None:
                self._limit_sem[self._process_queue[job_id].priority].release()

                logger.debug("{}: self._limit_sem[{}].release()".format(self._name,
                                                                        self._process_queue[job_id].priority))

            elif loop is not None and loop.warn():
                logger.warning("{}: self._limit_sem[{}] is blocked.".format(self._name,
                                                                            self._process_queue[job_id].priority))

        elif loop is not None and loop.warn():
            logger.warning("{}: self._next_call_sem is blocked.".format(self._name))

    def get_next_task(self):

        def _next():
            with self._main_sem:
                for _prio in self._prio_levels:

                    order = np.random.permutation(len(self._priorities[_prio]))

                    for pos in order:

                        _job_id = self._priorities[_prio][pos]

                        if self._process_queue[_job_id].num_pending() > 0:
                            return _job_id, _prio

            return None, None

        def _get(_job_id):
            with self._main_sem:
                if self.has(_job_id):
                    return self._process_queue[_job_id].get()
                else:
                    return None

        if self._feeder_queue is not None:
            if self._feeder_queue.prio_above_pending(self.max_prio()):
                return None, None, JobQueue.quit
        else:
            if not self.wait(self._process_queue_sem, msg="{}:_process_queue_sem".format(self._name)):
                return None, None, JobQueue.quit

        loop = InfiniteLoop(100, "{}:{}:{}".format(self._name,
                                                   inspect.getframeinfo(inspect.currentframe()).filename,
                                                   inspect.getframeinfo(inspect.currentframe()).lineno))

        while loop():
            job_id, prio = _next()

            if job_id is None:
                return None, None, JobQueue.quit

            if self._limit_sem is None:

                task_info = _get(job_id)

                if loop.warn():
                    logger.warning("{}: job_id: {}, task_info: {}".format(self._name, job_id, task_info))

                if task_info is not None:

                    if self._verbose:
                        logger.info("{}: job_id: {} #prio {} jobs: {}".
                              format(self._name, job_id, prio, len(self._priorities[prio])))

                    return job_id, task_info, JobQueue.quit

            elif self._limit_sem[prio].acquire(timeout=1):

                logger.debug("{}: self._limit_sem[{}].acquire()".format(self._name, prio))

                task_info = _get(job_id)

                if task_info is None:
                    self._limit_sem[prio].release()
                else:
                    if self._verbose:
                        logger.info("{}: job_id: {} #prio {} jobs: {}".
                              format(self._name, job_id, prio, len(self._priorities[prio])))

                    return job_id, task_info, JobQueue.quit

            elif loop.warn():
                logger.warning("{}: _limit_sem blocked, job_id: {}, prio: {}".format(self._name, job_id, prio))

    @staticmethod
    def wait(sem=None, msg=None):

        loop = InfiniteLoop(100, "{}:{}".format(inspect.getframeinfo(inspect.currentframe()).filename,
                                                inspect.getframeinfo(inspect.currentframe()).lineno))
        while loop():
            if sem is not None and sem.acquire(timeout=1):
                return True
            elif loop.warn() and msg is not None:
                logger.warning("sem blocked: {}", msg)

            if msg is not None:
                logger.info(msg)

            if JobQueue.quit:
                return False

            if sem is None:
                return True
