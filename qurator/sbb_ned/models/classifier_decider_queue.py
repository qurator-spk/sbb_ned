import os
import logging
import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

from collections import OrderedDict

from qurator.sbb_ned.models.bert import model_predict_compare
from qurator.sbb_ned.models.decider import DeciderTask
from qurator.utils.parallel import run as prun

from .jobs import JobQueue
from qurator.sbb_ned.models.bert import get_device
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, BertConfig, BertForSequenceClassification)

logger = logging.getLogger(__name__)


class DeciderTaskWrapper(DeciderTask):

    def __init__(self, job_id, **kwargs):

        self._job_id = job_id

        super(DeciderTaskWrapper, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):

        ret = self._job_id, super(DeciderTaskWrapper, self).__call__(*args, **kwargs)

        return ret


class ClassifierTask:

    model = None
    device = None
    batch_size = None

    def __init__(self, job_id, entity_id, features, candidates, **kwargs):

        self._job_id = job_id
        self._entity_id = entity_id
        self._features = features
        self._candidates = candidates

    def __call__(self, *args, **kwargs):

        if self._candidates is None:
            return self._job_id, None, None, None

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

        return self._job_id, self._entity_id, decision, self._candidates

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

    def __init__(self, no_cuda, model_dir, model_file, decider, entities, decider_processes,
                 classifier_processes, batch_size):

        self._no_cuda = no_cuda
        self._model_dir = model_dir
        self._model_file = model_file

        self._decider = decider

        self._entities = entities
        self._decider_processes = decider_processes
        self._classifier_processes = classifier_processes
        self._batch_size = batch_size

        self._rank_intervalls = np.linspace(0.001, 0.1, 100)
        self._quantiles = np.linspace(0.1, 1, 10)

        self._queue_classifier = JobQueue(result_sequence=self.infinite_process_sequence(),
                                          name="ClassifierDeciderQueue_classifier", min_level=2, verbose=True)

        self._queue_decider = JobQueue(name="ClassifierDeciderQueue_decider", min_level=2,
                                       feeder_queue=self._queue_classifier)

        self._queue_final_output = JobQueue(name="ClassifierDeciderQueue_final_output", min_level=2,
                                            feeder_queue=self._queue_decider)

    def run(self, job_sequence, len_sequence, return_full, threshold, priority):

        params = {'threshold': threshold, 'return_full': return_full}

        job_main = self._queue_classifier.add_job(task_info=job_sequence, task_len=len_sequence, priority=priority,
                                                  params=params)

        job_decider = self._queue_decider.add_job([], priority=priority, params=params)

        job_final_output = self._queue_final_output.add_job([], priority=priority, params=params)

        complete_result = OrderedDict()

        for job_id, eid, result in job_main.sequence():

            print('ClassifierDeciderQueue.run: {}:{}'.format(job_id, eid))

            complete_result[eid] = result

        job_main.remove()

        job_decider.remove()

        job_final_output.remove()

        return complete_result

    def infinite_process_sequence(self):

        for job_id, (eid, result) in \
                prun(self.get_decider_tasks(), initializer=DeciderTask.initialize,
                     initargs=(self._decider, self._entities), processes=self._decider_processes):

                self._queue_final_output.add_to_job(job_id, (eid, result))

                while True:
                    _, task_info, iter_quit = self._queue_final_output.get_next_task()

                    if iter_quit:
                        return

                    if task_info is None:
                        break

                    eid, result, params = task_info

                    yield job_id, (eid, result)

    def get_classifier_tasks(self):

        while True:

            job_id, task_info, iter_quit = self._queue_classifier.get_next_task()

            if iter_quit:
                return

            if job_id is None or task_info is None:
                continue

            _, entity_id, features, candidates, params = task_info

            print("get_classifier_tasks: {}:{}".format(job_id, entity_id))

            yield ClassifierTask(job_id, entity_id, features, candidates, **params)

    def get_decider_tasks(self):

        for job_id, entity_id, decision, candidates in \
                prun(self.get_classifier_tasks(), initializer=ClassifierTask.initialize,
                     initargs=(self._no_cuda, self._model_dir, self._model_file, self._batch_size),
                     processes=self._classifier_processes):

            self._queue_decider.add_to_job(job_id, (entity_id, decision, candidates))

            while True:
                _, task_info, iter_quit = self._queue_decider.get_next_task()

                if iter_quit:
                    return

                if task_info is None:
                    break

                entity_id, decision, candidates, params = task_info

                if entity_id is None:
                    continue
                if candidates is None:
                    continue

                yield DeciderTaskWrapper(job_id, entity_id=entity_id, decision=decision, candidates=candidates,
                                         quantiles=self._quantiles, rank_intervalls=self._rank_intervalls,
                                         **params)
