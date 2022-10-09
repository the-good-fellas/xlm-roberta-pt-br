from huggingface_hub import HfApi
from wasabi import Printer
import os


def brwac_parser(args):
  sent_flush_iter = 0
  sents = []
  sent_tokens = []
  api = HfApi()
  msg = Printer()
  train_file = open(f'brwac_train_{sent_flush_iter}.jsonl', 'a')
  eval_file = open(f'brwac_validation_{sent_flush_iter}.jsonl', 'a')
  test_file = open(f'brwac_test_{sent_flush_iter}.jsonl', 'a')

  train_file.write('text\n')
  eval_file.write('text\n')
  test_file.write('text\n')

  file_pointer = {
    'test': test_file,
    'eval': eval_file,
    'train': train_file
  }

  with open(args.brwac_file, 'r') as brwac:
    for idx, line in enumerate(brwac):
      if line.startswith('<p>') or \
        line.startswith('</p>') or \
        line.startswith('<g/>') or \
        line.startswith('</doc>') or \
        line.startswith('\n\n'):
        continue

      if line.startswith('<doc'):
        if len(sents) == 100_000:
          if sent_flush_iter == 0:
            f = file_pointer['test']
          elif sent_flush_iter == 1:
            f = file_pointer['eval']
          else:
            f = file_pointer['train']

          for s in sents:
            f.write(f'{s}\n')

          msg.info(f'uploading {f.name}')
          api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo=f.name,
            repo_type='dataset',
            repo_id=args.repo_id,
            token=args.hf_token
          )

          os.remove(f.name)

          sent_flush_iter += 1
          train_file = open(f'brwac_train_{sent_flush_iter}.jsonl', 'a')
          file_pointer['train'] = train_file

      if line.startswith('<s>'):
        sent_tokens = []
        continue

      if line.startswith('</s>'):
        sent = ' '.join(sent_tokens)
        sents.append(sent.lower())
        continue