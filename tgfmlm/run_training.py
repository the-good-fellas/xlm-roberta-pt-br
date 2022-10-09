from datasets import load_dataset
from transformers import (
  TrainingArguments,
  set_seed,
  XLMRobertaTokenizerFast,
  DataCollatorForLanguageModeling
)
import numpy as np
import wandb


def mlm_training(args):
  w_run = wandb.init(
    project=args.wandb_project,
    entity=args.wandb_run_id,
    id=args.wandb_entity
  )

  set_seed(42)

  # dataset preparation
  dataset = load_dataset(args.dataset_id)
