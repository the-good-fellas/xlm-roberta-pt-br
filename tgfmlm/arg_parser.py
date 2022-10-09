import argparse


class TGFArgs:
  def __init__(self):
    parser = argparse.ArgumentParser(description='TGF')
    parser.add_argument('-mode', default="mlm", type=str)
    parser.add_argument('--assets_path', default="assets", type=str)
    parser.add_argument('--base_model', default="xlm-roberta-base", type=str)
    parser.add_argument('--dataset_id', default="goodfellas/brwac_tiny", type=str)
    parser.add_argument('--wandb_project', default="tgf-xlm-roberta-base-pt-br", type=str)
    parser.add_argument('--wandb_run_id', default="tgf-xlm-roberta-base-pt-br-v1.0.0", type=str)
    parser.add_argument('--wandb_entity', default="tgf", type=str)
    parser.add_argument('--brwac_file', default="assets/brwac.vert", type=str)
    parser.add_argument('--repo_id', default="goodfellas/brwac", type=str)
    parser.add_argument('--hf_token', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int)
    parser.add_argument('--lr', default=5e-6, type=int)
    parser.add_argument('--max_length', default=252, type=int)

    self.opts = parser.parse_args()

  def get_params(self):
    return self.opts
