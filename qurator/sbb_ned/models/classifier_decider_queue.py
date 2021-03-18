import os
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
from qurator.sbb_ned.models.bert import get_device
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, BertConfig, BertForSequenceClassification)

logger = logging.getLogger(__name__)


class ClassifierTask:

    model = None
    device = None
    batch_size = None

    def __init__(self, entity_id, features, candidates):

        self._entity_id = entity_id
        self._features = features
        self._candidates = candidates

    def __call__(self, *args, **kwargs):

        all_input_ids = torch.tensor([f.input_ids for f in self._features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in self._features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in self._features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in self._features], dtype=torch.long)

        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)

        sampler = SequentialSampler(data)

        data_loader = DataLoader(data, sampler=sampler, batch_size=ClassifierTask.batch_size)

        decision = model_predict_compare(data_loader, ClassifierTask.device, ClassifierTask.model, disable_output=True)

        decision['guessed_title'] = [f.guid[1] for f in self._features]
        decision['target'] = [f.guid[0] for f in self._features]
        decision['scores'] = np.log(decision[1] / decision[0])

        assert len(decision.target.unique()) == 1

        return self._entity_id, decision, self._candidates

    @staticmethod
    def initialize(no_cuda, model_dir, model_file, batch_size):

        ClassifierTask.batch_size = batch_size

        ClassifierTask.device, n_gpu = get_device(no_cuda=no_cuda)

        config_file = os.path.join(model_dir, CONFIG_NAME)
        #
        model_file = os.path.join(model_dir, model_file)

        config = BertConfig(config_file)

        ClassifierTask.model = BertForSequenceClassification(config, num_labels=2)
        # noinspection PyUnresolvedReferences
        ClassifierTask.model.load_state_dict(torch.load(model_file,
                                                        map_location=lambda storage, loc: storage if no_cuda else None))
        # noinspection PyUnresolvedReferences
        ClassifierTask.model.to(ClassifierTask.device)
        # noinspection PyUnresolvedReferences
        ClassifierTask.model.eval()


class ClassifierDeciderQueue:

    quit = False

    def __init__(self, no_cuda, model_dir, model_file, decider, threshold, entities, decider_processes,
                 classifier_processes, batch_size):

        # logger.info('ClassifierDeciderQueue __init__')

        self._process_queue = []
        self._process_queue_sem = Semaphore(0)
        self._main_sem = Semaphore(1)

        self._no_cuda = no_cuda
        self._model_dir = model_dir
        self._model_file = model_file

        # self._model = model
        # self._device = device

        self._decider = decider
        self._threshold = threshold

        self._entities = entities
        self._decider_processes = decider_processes
        self._classifier_processes = classifier_processes
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

        print('\n\n\n\nDone2.\n\n\n\n')

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

    @staticmethod
    def get_classifier_tasks(job_sequence):

        for entity_id, features, candidates in tqdm(job_sequence):

            print("get_classifier_tasks: {}".format(entity_id))

            if entity_id is None:
                continue

            if len(candidates) == 0:
                continue

            yield ClassifierTask(entity_id, features, candidates)

    def get_decider_tasks(self):

        while True:

            if not self.wait(self._process_queue_sem):
                return

            job_sequence, len_sequence = self._process_queue.pop()

            for entity_id, decision, candidates in tqdm(prun(self.get_classifier_tasks(job_sequence),
                                                             initializer=ClassifierTask.initialize,
                                                             initargs=(self._no_cuda, self._model_dir, self._model_file,
                                                                       self._batch_size),
                                                             processes=self._classifier_processes),
                                                        total=len_sequence):

                yield DeciderTask(entity_id, decision, candidates, self._quantiles, self._rank_intervalls,
                                  self._threshold, self._return_full)

            # for entity_id, features, candidates in tqdm(job_sequence, total=len_sequence):
            #
            #     logger.debug("get_decider_tasks: {}".format(entity_id))
            #
            #     if entity_id is None:
            #         continue
            #
            #     if len(candidates) == 0:
            #         continue
            #
            #     all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            #     all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            #     all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            #     all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            #
            #     data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)
            #
            #     sampler = SequentialSampler(data)
            #
            #     data_loader = DataLoader(data, sampler=sampler, batch_size=self._batch_size)
            #
            #     decision = model_predict_compare(data_loader, self._device, self._model, disable_output=True)
            #     decision['guessed_title'] = [f.guid[1] for f in features]
            #     decision['target'] = [f.guid[0] for f in features]
            #     decision['scores'] = np.log(decision[1] / decision[0])
            #
            #     assert len(decision.target.unique()) == 1
            #
            #     yield DeciderTask(entity_id, decision, candidates, self._quantiles, self._rank_intervalls,
            #                       self._threshold, self._return_full)
            #
            # # signal to process_queue to return completed result
            # yield DeciderTask(entity_id=None, decision=None, candidates=None, quantiles=None, rank_intervalls=None,
            #                   threshold=None)

    @staticmethod
    def wait(sem=None):

        while True:
            if sem is not None and sem.acquire(timeout=10):
                return True

            if ClassifierDeciderQueue.quit:
                return False

            if sem is None:
                return True
