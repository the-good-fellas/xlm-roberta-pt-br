# Instruções de uso do repositório

É recomendado que se crie um ambiente exclusivo para seus experimentos, desta forma, se algo der errado as outras 
coisas continuarão funcionando, sendo assim, se estiver utilizando `anaconda`:

`conda create -n mlm python=3.9`

`conda activate mlm`

### Execução

Para executar o script use:

```shell
nohup python3 -m tgfmlm -mode mlm \
--output_dir /thegoodfellas/model \
--base_model xlm-roberta-base \
--dataset_id thegoodfellas/brwac_sentences \
--wandb_project tgf-xlm-roberta-base-pt-br \
--wandb_run_id tgf-xlm-roberta-base-pt-br-v1.2 \
--wandb_entity thegoodfellas \
--epochs 2 \
--batch_size 16 \
--gradient_accumulation_steps 8 \
--lr 1e-4 \
--logging_steps 10 \
--save_steps 200 \
--max_length 512 \
--warmup_steps 1_000 \
--num_proc 50 \
--hub_model_id thegoodfellas/tgf-xlm-roberta-base-pt-br &
```

Acompanhe os logs pelo WandB