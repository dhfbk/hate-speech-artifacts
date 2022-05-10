from typing import Dict, List, Any
from overrides import overrides
import logging
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.modules import InputVariationalDropout
from allennlp.modules import Embedding

from machamp.util import log_training_dynamics

logger = logging.getLogger(__name__)


@Model.register("machamp_model")
class MachampModel(Model):
    """
    """

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            encoder: Seq2VecEncoder,
            decoders: Dict[str, Model],
            tasks: List[str],
            task_types: List[str],
            dropout: float = None,
            dataset_embedder: Embedding = None,
            **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder

        self.encoder = encoder
        self._classifier_input_dim = self.encoder.get_output_dim()

        if dropout:
            # TODO: is the variational dropout better (for xlm?)
            self._dropout = InputVariationalDropout(dropout)
            self._dropout_sents = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self.decoders = torch.nn.ModuleDict(decoders)

        self.tasks = tasks
        self.task_types = task_types

        self.counter = 0
        self.metrics = {}
        self.no_dev = False
        self.dataset_embedder = dataset_embedder
        if self.dataset_embedder == None and vocab.get_vocab_size('dataset_embeds') > 0:
            # TODO get 768 automatically
            self.dataset_embedder = Embedding(768, vocab.get_vocab_size("dataset_embeds"))


    def forward(self,
                tokens: TextFieldTensors,
                dataset=None,
                metadata: List[Dict[str, Any]] = None,
                **kwargs: Dict[str, torch.LongTensor]
                ) -> Dict[str, torch.Tensor]:
        """
        """
        #print(tokens['tokens']['type_ids'].shape)
        #print(tokens['tokens'])
        #for item in tokens['tokens']:
        #    print(item, tokens['tokens'][item].shape)
        #print()
        #tokens['tokens']['type_ids'] = tokens['tokens']['mask'].to(torch.int)
        #for item in tokens['tokens']:
        #    print(item, tokens['tokens'][item].shape)
        #print(tokens['tokens'])
        #exit(1)
        #.shape, dtype=torch.long, device=tokens['tokens']['type_ids'].device)
        #print(tokens['tokens']['type_ids'].shape)

        #print(tokens['tokens']['type_ids'])
        #print(tokens['tokens']['type_ids'].shape)
        #tokens['tokens']['type_ids'] = torch.ones_like(tokens['tokens']['type_ids'])
        #print(tokens['tokens']['type_ids'])
        #print(tokens['tokens']['type_ids'].shape)
        #print()
        #print()
        

        gold_labels = kwargs 

        #print()
        #print(len(metadata[0]['wordpiece_sizes']))
        #print(gold_labels['dataset_embeds'][0])
        
        tasks_to_handle = []
        task_types_to_handle = []
        self.no_dev = metadata[0]['no_dev']
        self.is_train = metadata[0]['is_train']
        self.label_counts = metadata[0]['label_counts']

        if self.no_dev:
            return {}
        for task, task_type in zip(self.tasks, self.task_types):
            s2s_and_in = task_type == "seq2seq" and 'target_words' in metadata[0]['col_idxs']
            dep_and_in = task_type == 'dependency' and task + '_rels' in metadata[0]['col_idxs']
            if s2s_and_in or dep_and_in or task in metadata[0]['col_idxs']:
                tasks_to_handle.append(task)
                task_types_to_handle.append(task_type)

        sent_count = task_types_to_handle.count('classification')
        mask = get_text_field_mask(tokens)


        if 'dataset_embeds' in gold_labels:
            embedded_text = self._text_field_embedder(tokens, dataset_ids=gold_labels['dataset_embeds'], dataset_embedder=self.dataset_embedder)
        else:
            embedded_text = self._text_field_embedder(tokens)
    
        if sent_count > 0:
            embedded_text_sent = self.encoder(self._text_field_embedder(tokens), mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)
            if sent_count > 0:
                embedded_text_sent = self._dropout_sents(embedded_text_sent)

        logits = {}
        class_probabilities = {}
        training_dynamics_tasks = {}
        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities,
                       "log_training_dynamics": training_dynamics_tasks}
        loss = 0.0

        for task, task_type in zip(tasks_to_handle, task_types_to_handle):
            if task_type == 'classification':
                task_gold_labels = None if task not in gold_labels else gold_labels[task]
                # pred_output = self.decoders[task].forward(embedded_text_sent, task_gold_labels)
                pred_output = self.decoders[task].forward(embedded_text_sent, task_gold_labels, label_counts=self.label_counts[task])
                class_probabilities[task] = pred_output["class_probabilities"]
                training_dynamics_tasks[task] = pred_output["log_training_dynamics"]
            elif task_type == 'dependency':
                tags_gold_labels = None if task + '_rels' not in gold_labels else gold_labels[task + '_rels']
                indices_gold_labels = None if task + '_head_indices' not in gold_labels else gold_labels[task + '_head_indices']
                pred_output = self.decoders[task].forward(embedded_text, mask=mask,
                                                          gold_head_tags=tags_gold_labels,
                                                          gold_head_indices=indices_gold_labels)
                class_probabilities[task + '_rels'] = pred_output[task + "_rels"]
                class_probabilities[task + '_head_indices'] = pred_output[task + "_head_indices"]
                training_dynamics_tasks[task] = pred_output["log_training_dynamics"]
            elif task_type == 'mlm':
                task_gold_labels = None if task not in gold_labels else gold_labels[task]
                pred_output = self.decoders[task].forward(embedded_text, task_gold_labels)
                training_dynamics_tasks[task] = False # avoid computing dynamics for MLM by default
            elif task_type == 'seq2seq':
                task_gold_labels = None if 'target' not in gold_labels else gold_labels['target']
                pred_output = self.decoders[task].forward(embedded_text, mask, task_gold_labels)
                class_probabilities[task] = pred_output["class_probabilities"]
                training_dynamics_tasks[task] = False # avoid computing dynamics for seq2seq by default
            else:
                task_gold_labels = None if task not in gold_labels else gold_labels[task]
                pred_output = self.decoders[task].forward(embedded_text, task_gold_labels, mask=mask)
                class_probabilities[task] = pred_output["class_probabilities"]
                training_dynamics_tasks[task] = pred_output["log_training_dynamics"]

            if 'loss' in pred_output:
                logits[task] = pred_output['logits']

            dep_and_in = task_type == "dependency" and task + "_rels" in gold_labels

            s2s_and_in = task_type == "seq2seq" and 'target_words' in gold_labels
            if dep_and_in or s2s_and_in or task in gold_labels:
                loss += pred_output["loss"]

        if gold_labels:
            output_dict['loss'] = loss

        if metadata is not None:
            output_dict["tokens"] = [x["tokens"] for x in metadata]
            output_dict["full_data"] = [x['full_data'] for x in metadata]
            output_dict["col_idxs"] = [x['col_idxs'] for x in metadata]

            # Rob: Warning, hacky!, allennlp requires them to be in the length of metadata, in the dump_lines I just use the first
            output_dict['tasks'] = [self.tasks for _ in metadata]
            output_dict["task_types"] = [self.task_types for _ in metadata]
        output_dict['mask'] = mask

        # Only during training, log training dynamics for each supported task (if specified)
        if self.is_train:
            for task, task_type in zip(tasks_to_handle, task_types_to_handle):
                if task not in ["mlm", "seq2seq"]:
                    if output_dict["log_training_dynamics"][task]==True:
                        train_golds = gold_labels[task + '_rels'] if task == "dependency" else gold_labels[task]
                        # TODO: figure out why in "dependency" the dimension of train_golds is 3 and train_logits is 4 using the example in the repo
                        log_training_dynamics(output_dir=self.serialization_dir,
                                              epoch=self.epoch,
                                              task=task,
                                              train_ids=[m["instance_idx"] for m in metadata],
                                              train_logits=output_dict["logits"][task].detach().cpu().tolist(),
                                              train_golds=train_golds.detach().cpu().tolist())
                else:
                    logger.warning(f"WARNING. Training dynamics for task type: {task_type} are not supported due to its potentially large label space. Skipping.")

        return output_dict


    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        for task in self.tasks:
            is_dep = self.task_types[self.tasks.index(task)] == 'dependency'
            if task in output_dict['class_probabilities']: 
                output_dict[task] = self.decoders[task].make_output_human_readable(output_dict['class_probabilities'][task])
            elif is_dep and task + '_rels' in output_dict['class_probabilities']:
                dep_tags = output_dict['class_probabilities'][task + '_rels']
                dep_heads = output_dict['class_probabilities'][task + '_head_indices']
                mask = output_dict['mask']
                output_dict[task + '_rels'], output_dict[task + '_head_indices'] = \
                                    self.decoders[task].make_output_human_readable(dep_tags, dep_heads, mask)
        
        if 'loss' not in output_dict or output_dict['loss'] == 0:
            output_dict['loss'] = [output_dict['loss']]
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for task in self.tasks:
            for name, task_metric in self.decoders[task].get_metrics(reset).items():
                if name.split("/")[-1] in ["acc", 'perplexity']:
                    metrics[name] = task_metric
                elif name.split('/')[-1].lower() in ['las']:
                    metrics[name] = task_metric['LAS']
                elif name.split('/')[-1] in ['micro-f1', 'macro-f1']:
                    metrics[name] = task_metric['fscore']
                elif name.split("/")[-1] == "span_f1":
                    metrics[name] = task_metric["f1-measure-overall"]
                elif name.split("/")[-1] == "multi_span_f1":
                    metrics[name] = task_metric["f1-measure-overall"]
                elif name.split("/")[-1] == "bleu":
                    metrics[name] = task_metric["BLEU"]
                else:
                    logger.error(f"ERROR. Metric: {name} unrecognized.")
        # The "sum" metric summing all tracked metrics keeps a good measure of patience for early stopping and saving
        metrics_to_track = set()
        for task, task_type in zip(self.tasks, self.task_types):
            metrics_to_track.add(task if task_type != 'dependency' else 'las')

        metric_sum = 0.0
        for name, metric in metrics.items():
            if (not name.startswith("_") and set(name.split("/")).intersection(metrics_to_track)) or name=='.run/.counter':
                if name == '.run/.counter':
                    continue
                if name.endswith("perplexity"):
                    if metric != 0.0:
                        metric_sum += 1/metric
                else:
                    metric_sum += metric

        if self.no_dev and metric_sum == 0.0:
            self.counter+= 0.001
            metrics[".run/.counter"] = self.counter
            metric_sum = self.counter

        metrics[".run/.sum"] = metric_sum
        return metrics

