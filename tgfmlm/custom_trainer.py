from transformers import Trainer
from torch import nn
import torch


class TgfMlmTrainer(Trainer):

  def __init__(self, *args, w_run, **kwargs):
    super(TgfMlmTrainer, self).__init__(*args, **kwargs)
    self._w_run = w_run

  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.get("labels")
    # forward pass
    outputs = model(**inputs)
    logits = outputs.get('logits')
    # compute custom loss
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
    self._w_run.log({'perplexity': torch.exp(loss)})
    return (loss, outputs) if return_outputs else loss
