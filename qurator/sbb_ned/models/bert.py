from __future__ import absolute_import, division, print_function
# from inspect import currentframe

import logging
import os
import random
import json
import click

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME,  # WEIGHTS_NAME,
                                              BertConfig,
                                              BertForSequenceClassification)
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from qurator.sbb_ner.models.tokenization import BertTokenizer

from tqdm import tqdm, trange

from ..embeddings.base import load_embeddings
from ..ground_truth.data_processor import WikipediaNEDProcessor

# from sklearn.model_selection import GroupKFold

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def model_train(bert_model, max_seq_length, do_lower_case,
                num_train_epochs, train_batch_size, gradient_accumulation_steps,
                learning_rate, weight_decay, loss_scale, warmup_proportion,
                processor, device, n_gpu, fp16, cache_dir, local_rank,
                dry_run, no_cuda, output_dir=None, model_file=None):

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            gradient_accumulation_steps))

    train_batch_size = train_batch_size // gradient_accumulation_steps

    train_dataloader = processor.get_train_examples(train_batch_size, local_rank)

    # Batch sampler divides by batch_size!
    num_train_optimization_steps = int(len(train_dataloader) * num_train_epochs / gradient_accumulation_steps)

    if local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = cache_dir if cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                         'distributed_{}'.format(local_rank))

    model = BertForSequenceClassification.from_pretrained(bert_model, cache_dir=cache_dir,
                                                          num_labels=processor.num_labels())

    if fp16:
        model.half()

    model.to(device)

    if local_rank != -1:
        try:
            # noinspection PyPep8Naming
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)

        warmup_linear = WarmupLinearSchedule(warmup=warmup_proportion, t_total=num_train_optimization_steps)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate, warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)
        warmup_linear = None

    global_step = 0
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    logger.info("  Num epochs = %d", num_train_epochs)
    logger.info("  Target learning rate = %f", learning_rate)

    model_config = {"bert_model": bert_model, "do_lower": do_lower_case, "max_seq_length": max_seq_length}

    def save_model(lh):

        if output_dir is None:
            return

        if model_file is None:
            output_model_file = os.path.join(output_dir, "pytorch_model_ep{}.bin".format(ep))
        else:
            output_model_file = os.path.join(output_dir, model_file)

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        torch.save(model_to_save.state_dict(), output_model_file)

        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        json.dump(model_config, open(os.path.join(output_dir, "model_config.json"), "w"))

        lh = pd.DataFrame(lh, columns=['global_step', 'loss'])

        loss_history_file = os.path.join(output_dir, "loss_ep{}.pkl".format(ep))

        lh.to_pickle(loss_history_file)

    def load_model(epoch):

        if output_dir is None:
            return False

        if model_file is None:
            output_model_file = os.path.join(output_dir, "pytorch_model_ep{}.bin".format(epoch))
        else:
            output_model_file = os.path.join(output_dir, model_file)

        if not os.path.exists(output_model_file):
            return False

        logger.info("Loading epoch {} from disk...".format(epoch))
        model.load_state_dict(torch.load(output_model_file,
                                         map_location=lambda storage, loc: storage if no_cuda else None))
        return True

    model.train()
    for ep in trange(1, int(num_train_epochs) + 1, desc="Epoch"):

        if dry_run and ep > 1:
            logger.info("Dry run. Stop.")
            break

        if model_file is None and load_model(ep):

            global_step += len(train_dataloader) // gradient_accumulation_steps
            continue

        loss_history = list()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {ep}") as pbar:

            for step, batch in enumerate(train_dataloader):

                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, labels = batch

                loss = model(input_ids, segment_ids, input_mask, labels)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                loss_history.append((global_step, loss.item()))

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                pbar.update(1)
                mean_loss = tr_loss * gradient_accumulation_steps / nb_tr_steps
                pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")

                if dry_run and len(loss_history) > 2:
                    logger.info("Dry run. Stop.")
                    break

                if (step + 1) % gradient_accumulation_steps == 0:
                    if fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = learning_rate * warmup_linear.get_lr(global_step, warmup_proportion)

                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        save_model(loss_history)

    return model, model_config


def model_eval(batch_size, processor, device, num_train_epochs=1, output_dir=None, model=None,
               local_rank=-1, no_cuda=False, dry_run=False, model_file=None):

    output_eval_file = None
    if output_dir is not None:
        output_eval_file = os.path.join(output_dir, processor.get_evaluation_file())
        logger.info('Write evaluation results to: {}'.format(output_eval_file))

    dataloader = processor.get_dev_examples(batch_size, local_rank)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataloader))
    logger.info("  Batch size = %d", batch_size)

    if output_dir is not None:
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        if not os.path.exists(output_config_file):
            raise RuntimeError("Cannot find model configuration file {}.".format(output_config_file))

        config = BertConfig(output_config_file)
    else:
        raise RuntimeError("Cannot find model configuration file. Output directory is missing.")

    model = None

    def load_model(epoch):

        nonlocal model

        if output_dir is None:
            return False

        if model_file is None:
            output_model_file = os.path.join(output_dir, "pytorch_model_ep{}.bin".format(epoch))
        else:
            output_model_file = os.path.join(output_dir, model_file)

        if not os.path.exists(output_model_file):
            logger.info("Stopping at epoch {} since model file is missing ({}).".format(ep, output_model_file))
            return False

        logger.info("Loading epoch {} from disk...".format(epoch))
        model = BertForSequenceClassification(config, num_labels=processor.num_labels())

        # noinspection PyUnresolvedReferences
        model.load_state_dict(torch.load(output_model_file,
                                         map_location=lambda storage, loc: storage if no_cuda else None))

        # noinspection PyUnresolvedReferences
        model.to(device)

        return True

    results = []
    for ep in trange(1, int(num_train_epochs) + 1, desc="Epoch"):

        if dry_run and ep > 1:
            logger.info("Dry run. Stop.")
            break

        if not load_model(ep):
            break

        if model is None:
            raise ValueError('Model required for evaluation.')

        # noinspection PyUnresolvedReferences
        model.eval()

        results.append(model_predict_compare(dataloader, device, model))

        if output_eval_file is not None:
            pd.concat(results).to_pickle(output_eval_file)


def model_predict_compare(dataloader, device, model, disable_output=False):

    decision_values = list()
    for input_ids, input_mask, segment_ids, labels in tqdm(dataloader, desc="Evaluating", total=len(dataloader),
                                                           disable=disable_output):

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        tmp = pd.DataFrame(F.softmax(logits, dim=1).cpu().numpy())
        tmp['labels'] = labels.cpu().numpy()

        decision_values.append(tmp)

    return pd.concat(decision_values).reset_index(drop=True)


def get_device(local_rank=-1, no_cuda=False):
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    return device, n_gpu


@click.command()
@click.argument("bert-model", type=str, required=True, nargs=1)
@click.argument("output-dir", type=str, required=True, nargs=1)
@click.option("--model-file", type=click.Path(), default=None, help="")
@click.option("--train-set-file", type=click.Path(exists=True), default=None, help="")
@click.option("--dev-set-file", type=click.Path(exists=True), default=None, help="")
@click.option("--test-set-file", type=click.Path(exists=True), default=None, help="")
@click.option("--train-size", default=0, type=int, help="")
@click.option("--dev-size", default=0, type=int, help="")
@click.option("--train-size", default=0, type=int, help="")
@click.option("--cache-dir", type=click.Path(), default=None,
              help="Where do you want to store the pre-trained models downloaded from s3")
@click.option("--max-seq-length", default=128, type=int,
              help="The maximum total input sequence length after WordPiece tokenization. \n"
                   "Sequences longer than this will be truncated, and sequences shorter \n than this will be padded.")
@click.option("--do-lower-case", is_flag=True, help="Set this flag if you are using an uncased model.", default=False)
@click.option("--train-batch-size", default=32, type=int, help="Total batch size for training.")
@click.option("--eval-batch-size", default=8, type=int, help="Total batch size for eval.")
@click.option("--learning-rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
@click.option("--weight-decay", default=0.01, type=float, help="Weight decay for Adam.")
@click.option("--num-train-epochs", default=3.0, type=float, help="Total number of training epochs to perform/evaluate.")
@click.option("--warmup-proportion", default=0.1, type=float,
              help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
@click.option("--no-cuda", is_flag=True, help="Whether not to use CUDA when available", default=False)
@click.option("--dry-run", is_flag=True, default=False, help="Test mode.")
@click.option("--local-rank", type=int, default=-1, help="local_rank for distributed training on gpus")
@click.option('--seed', type=int, default=42, help="random seed for initialization")
@click.option('--gradient-accumulation-steps', type=int, default=1,
              help="Number of updates steps to accumulate before performing a backward/update pass. default: 1")
@click.option('--fp16', is_flag=True, default=False, help="Whether to use 16-bit float precision instead of 32-bit")
@click.option('--loss-scale', type=float, default=0.0,
              help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n "
                   "0 (default value): dynamic loss scaling.\n"
                   "Positive power of 2: static loss scaling value.\n")
@click.option("--ned-sql-file", type=click.Path(exists=True), default=None, required=False)
@click.option('--embedding-type', type=click.Choice(['fasttext']), default='fasttext')
@click.option('--embedding-model', type=click.Path(exists=True), default=None)
@click.option('--n-trees', type=int, default=100)
@click.option('--distance-measure', type=click.Choice(['angular', 'euclidean']), default='angular')
@click.option('--entity-index-path', type=click.Path(exists=True), default=None)
@click.option('--entities-file', type=click.Path(exists=True), default=None)
def main(bert_model, output_dir,
         train_set_file, dev_set_file, test_set_file,  cache_dir, max_seq_length,
         train_size=0, dev_size=0, test_size=0,
         do_lower_case=False, train_batch_size=32, eval_batch_size=8, learning_rate=3e-5,
         weight_decay=0.01, num_train_epochs=3, warmup_proportion=0.1, no_cuda=False, dry_run=False, local_rank=-1,
         seed=42, gradient_accumulation_steps=1, fp16=False, loss_scale=0.0,
         ned_sql_file=None, search_k=50, max_dist=0.25, embedding_type='fasttext', embedding_model=None, n_trees=100,
         distance_measure='angular', entity_index_path=None, entities_file=None, model_file=None):
    """
    ned_sql_file: \n
    train_set_file: \n
    dev_set_file: \n
    test_set_file: \n
    bert_model: Bert pre-trained model selected in the list:\n
                bert-base-uncased, bert-large-uncased, bert-base-cased,\n
                bert-large-cased, bert-base-multilingual-uncased,\n
                bert-base-multilingual-cased, bert-base-chinese.\n
    output_dir: The output directory where the model predictions
                and checkpoints will be written.\n
    """

    device, n_gpu = get_device(local_rank, no_cuda)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(local_rank != -1), fp16))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not train_size > 0 and not dev_size > 0:
        raise ValueError("At least one of train_size or dev_size must be > 0.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if ned_sql_file is not None:

        if entity_index_path is None:
            raise RuntimeError("entity-index-path required!")

        if entities_file is None:
            raise RuntimeError("entities-file required!")

        embs = load_embeddings(embedding_type, model_path=embedding_model)

        embeddings = {'PER': embs, 'LOC': embs, 'ORG': embs}

        train_subset = pd.read_pickle(train_set_file)
        dev_subset = pd.read_pickle(dev_set_file)
        test_subset = pd.read_pickle(test_set_file)

        processor_args = {'train_subset': train_subset,
                          'dev_subset': dev_subset,
                          'test_subset': test_subset,
                          'train_size': train_size,
                          'dev_size': dev_size,
                          'test_size': test_size,
                          'ned_sql_file': ned_sql_file,
                          'max_seq_length': max_seq_length,
                          'entities_file': entities_file,
                          'embeddings': embeddings,
                          'n_trees': n_trees,
                          'distance_measure': distance_measure,
                          'entity_index_path': entity_index_path,
                          'search_k': search_k,
                          'max_dist': max_dist,
                          'bad_count': 10,
                          'lookup_processes': 4,
                          'pairing_processes': 20}

        processor_class = WikipediaNEDProcessor
    else:
        raise RuntimeError("Do not know what processor to use.")

    if train_size > 0:
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

        with processor_class(tokenizer=tokenizer, **processor_args) as processor:

            model_train(bert_model=bert_model, output_dir=output_dir, max_seq_length=max_seq_length,
                        do_lower_case=do_lower_case, num_train_epochs=num_train_epochs,
                        train_batch_size=train_batch_size,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        learning_rate=learning_rate, weight_decay=weight_decay, loss_scale=loss_scale,
                        warmup_proportion=warmup_proportion, processor=processor, device=device, n_gpu=n_gpu,
                        fp16=fp16, cache_dir=cache_dir, local_rank=local_rank, dry_run=dry_run,
                        no_cuda=no_cuda, model_file=model_file)

    # noinspection PyUnresolvedReferences
    if dev_size > 0 and (local_rank == -1 or torch.distributed.get_rank() == 0):

        model_config = json.load(open(os.path.join(output_dir, "model_config.json"), "r"))

        tokenizer = BertTokenizer.from_pretrained(model_config['bert_model'],
                                                  do_lower_case=model_config['do_lower'])

        with processor_class(tokenizer=tokenizer, **processor_args) as processor:

            model_eval(processor=processor, device=device, num_train_epochs=num_train_epochs,
                       output_dir=output_dir, batch_size=eval_batch_size, local_rank=local_rank,
                       no_cuda=no_cuda, dry_run=dry_run, model_file=model_file)


if __name__ == "__main__":
    main()
