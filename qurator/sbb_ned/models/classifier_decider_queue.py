import logging
import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

from collections import OrderedDict

from qurator.sbb_ned.models.bert import model_predict_compare
from qurator.sbb_ned.models.decider import DeciderTask
from qurator.utils.parallel import run as prun

from tqdm import tqdm as tqdm
from multiprocessing import Semaphore

logger = logging.getLogger(__name__)


class ClassifierDeciderQueue:

    quit = False

    def __init__(self, model, device, decider, threshold, entities, decider_processes, batch_size):

        logger.info('ClassifierDeciderQueue __init__')

        self._process_queue = []
        self._process_queue_sem = Semaphore(0)
        self._main_sem = Semaphore(1)

        self._model = model
        self._device = device

        self._decider = decider
        self._threshold = threshold

        self._entities = entities
        self._decider_processes = decider_processes
        self._batch_size = batch_size

        self._rank_intervalls = np.linspace(0.001, 0.1, 100)
        self._quantiles = np.linspace(0.1, 1, 10)
        self._return_full = False

        self._sequence = self.process_sequence()

    def run(self, job_sequence, len_sequence, return_full, threshold):

        self._main_sem.acquire()

        _threshold, self._threshold = self._threshold, threshold
        _return_full, self._return_full = self._return_full, return_full

        self._process_queue.append((job_sequence, len_sequence))

        self._process_queue_sem.release()

        ret = next(self._sequence)

        self._threshold, self._return_full = _threshold, _return_full

        self._main_sem.release()

        return ret

    def process_sequence(self):

        complete_result = OrderedDict()

        for eid, result in prun(self.get_decider_tasks(), initializer=DeciderTask.initialize,
                                initargs=(self._decider, self._entities),
                                processes=self._decider_processes):

            if eid is None:
                yield complete_result
                complete_result = OrderedDict()
                continue

            if result is None:
                continue

            complete_result[eid] = result

    def get_decider_tasks(self):

        while True:

            if not self.wait(self._process_queue_sem):
                return

            job_sequence, len_sequence = self._process_queue.pop()

            for entity_id, features, candidates in tqdm(job_sequence, total=len_sequence):

                logger.debug("get_decider_tasks: {}".format(entity_id))

                if entity_id is None:
                    continue

                if len(candidates) == 0:
                    continue

                all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

                data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)

                sampler = SequentialSampler(data)

                data_loader = DataLoader(data, sampler=sampler, batch_size=self._batch_size)

                decision = model_predict_compare(data_loader, self._device, self._model, disable_output=True)
                decision['guessed_title'] = [f.guid[1] for f in features]
                decision['target'] = [f.guid[0] for f in features]
                decision['scores'] = np.log(decision[1] / decision[0])

                assert len(decision.target.unique()) == 1

                yield DeciderTask(entity_id, decision, candidates, self._quantiles, self._rank_intervalls,
                                  self._threshold, self._return_full)

            # signal to process_queue to return completed result
            yield DeciderTask(entity_id=None, decision=None, candidates=None, quantiles=None, rank_intervalls=None,
                              threshold=None)

    @staticmethod
    def wait(sem=None):

        while True:
            if sem is not None and sem.acquire(timeout=10):
                return True

            if ClassifierDeciderQueue.quit:
                return False

            if sem is None:
                return True
