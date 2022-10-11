import argparse


class TGFArgs:
  def __init__(self):
    parser = argparse.ArgumentParser(description='TGF')
    parser.add_argument('-mode', default="mlm", type=str)
    parser.add_argument('--assets_path', default="/thegoodfellas/assets", type=str)
    parser.add_argument('--output_dir', default="/thegoodfellas/model", type=str)
    parser.add_argument('--base_model', default="xlm-roberta-base", type=str)
    parser.add_argument('--dataset_id', default="goodfellas/brwac_tiny", type=str)
    parser.add_argument('--wandb_project', default="tgf-xlm-roberta-base-pt-br", type=str)
    parser.add_argument('--wandb_run_id', default="tgf-xlm-roberta-base-pt-br-dummy", type=str)
    parser.add_argument('--wandb_entity', default="thegoodfellas", type=str)
    parser.add_argument('--brwac_file', default="brwac.vert", type=str)
    parser.add_argument('--hf_token', type=str)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=int)
    parser.add_argument('--logging_steps', default=1, type=int)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--warmup_steps', default=10000, type=int)
    parser.add_argument('--hub_model_id', default='tgf-xlm-roberta-base-pt-br', type=str)

    self.opts = parser.parse_args()

  def get_params(self):
    return self.opts
