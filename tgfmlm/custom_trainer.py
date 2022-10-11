from transformers import Trainer
from typing import Dict
import torch


class TgfMlmTrainer(Trainer):

  def __init__(self, *args, w_run, **kwargs):
    self._w_run = w_run
    super(TgfMlmTrainer, self).__init__(*args, **kwargs)

  def log(self, logs: Dict[str, float]) -> None:
    logs["learning_rate"] = self._get_learning_rate()
    super().log(logs)

  def compute_loss(self, model, inputs, return_outputs=False):
    # forward pass
    outputs = model(**inputs)
    loss = outputs.get('loss')
    self._w_run.log({'ppl': torch.exp(loss)})
    return outputs if return_outputs else loss
