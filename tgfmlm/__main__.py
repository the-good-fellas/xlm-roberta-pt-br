from tgfmlm.run_training import mlm_training
from tgfmlm.brwac_to_hf import brwac_parser
from tgfmlm.arg_parser import TGFArgs

if __name__ == '__main__':
  args = TGFArgs().get_params()
  if args.mode == 'mlm':
    mlm_training(args)
  elif args.mode == 'brwac':
    brwac_parser(args)
  else:
    raise NotImplemented('use mlm or brwac')
