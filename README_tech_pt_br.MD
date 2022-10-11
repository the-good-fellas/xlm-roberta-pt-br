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
--dataset_id thegoodfellas/brwac_tiny \
--wandb_project tgf-xlm-roberta-base-pt-br \
--wandb_run_id tgf-xlm-roberta-base-pt-br-dummy \
--wandb_entity thegoodfellas \
--epochs 2 \
--batch_size 64 \
--gradient_accumulation_steps 1 \
--lr 1e-4 \
--logging_steps 5000 \
--max_length 512 \
--warmup_steps 10_000 \
--hub_model_id thegoodfellas/tgf-xlm-roberta-base-pt-br &
```

Acompanhe os logs pelo WandB