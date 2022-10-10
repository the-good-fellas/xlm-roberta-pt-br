from transformers.training_args import HubStrategy
from datasets import load_dataset
from transformers import (
  TrainingArguments,
  Trainer,
  set_seed,
  XLMRobertaTokenizerFast,
  XLMRobertaConfig,
  XLMRobertaForMaskedLM,
  DataCollatorForLanguageModeling
)
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
    num_proc=5,
    remove_columns=['text'],
    load_from_cache_file=True,
  )

  data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

  # training
  training_args = TrainingArguments(
    output_dir=args.model_name,
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=True,
    save_steps=10_000,
    save_total_limit=1,
    warmup_steps=args.warmup_steps,
    weight_decay=0.01,
    learning_rate=args.lr,
    report_to=["wandb"],
    logging_steps=1_000,
    do_eval=False,
    push_to_hub=True,
    hub_strategy=HubStrategy.CHECKPOINT,
    fp16=True
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds_train,
    tokenizer=tokenizer,
    data_collator=data_collator,
    w_run=w_run
  )

  trainer.train()
  trainer.save_model()
  w_run.finish()
