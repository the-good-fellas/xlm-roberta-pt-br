from transformers.training_args import HubStrategy
from tgfmlm.custom_trainer import TgfMlmTrainer
from datasets import load_dataset
from transformers import (
  IntervalStrategy,
  TrainingArguments,
  set_seed,
  XLMRobertaTokenizerFast,
  XLMRobertaConfig,
  XLMRobertaForMaskedLM,
  DataCollatorForLanguageModeling
)
import wandb
import torch
import os


def mlm_training(args):
  os.makedirs(args.output_dir, exist_ok=True)

  set_seed(42)

  # dataset preparation
  dataset = load_dataset(args.dataset_id)

  # model/tokenizer init
  config = XLMRobertaConfig.from_pretrained(args.base_model)
  tokenizer = XLMRobertaTokenizerFast.from_pretrained(
    args.base_model,
    max_len=512,
    add_prefix_space=True)
  model = XLMRobertaForMaskedLM.from_pretrained(
    args.base_model,
    config=config
  )

  def tokenize_function(examples):
    return tokenizer(
      examples["text"],
      padding=True,
      truncation=True,
      max_length=args.max_length,
      return_special_tokens_mask=True
    )

  tokenized_ds_train = dataset['train'].map(
    tokenize_function,
    batched=True,
    num_proc=args.num_proc,
    remove_columns=['text'],
    load_from_cache_file=True,
  )

  data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

  # training
  training_args = TrainingArguments(
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=True,
    save_strategy=IntervalStrategy.STEPS,
    save_steps=args.save_steps,
    save_total_limit=1,
    warmup_steps=args.warmup_steps,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    learning_rate=args.lr,
    report_to=["wandb"],
    logging_steps=args.logging_steps,
    do_eval=False,
    push_to_hub=True,
    hub_strategy=HubStrategy.CHECKPOINT,
    hub_model_id=args.hub_model_id,
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    resume_from_checkpoint=True,
    fp16=True
  )

  w_run = wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    id=args.wandb_run_id
  )

  trainer = TgfMlmTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds_train,
    tokenizer=tokenizer,
    data_collator=data_collator,
    w_run=w_run
  )

  torch.cuda.empty_cache()
  # if args.resume:
  trainer.train(f'{args.output_dir}/last-checkpoint')
  # else:
  #   trainer.train()

  trainer.save_model()
  w_run.finish()
