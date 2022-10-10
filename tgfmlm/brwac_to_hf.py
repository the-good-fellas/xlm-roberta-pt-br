from huggingface_hub import HfApi
from wasabi import Printer
import os


def brwac_parser(args):
  sent_flush_iter = 0
  sents = []
  sent_tokens = []
  api = HfApi()
  msg = Printer()
  train_file = open(f'{args.assets_path}/brwac_train_{sent_flush_iter}.txt', 'a')

  train_file.write('text\n')

  msg.info('creating brwac dataset')
  with open(args.brwac_file, 'r') as brwac:
    for idx, line in enumerate(brwac):
      if line.startswith('<p>') or \
        line.startswith('</p>') or \
        line.startswith('<g/>') or \
        line.startswith('</doc>') or \
        line.startswith('\n\n'):
        continue

      if line.startswith('<doc'):
        if len(sents) >= 100_000:

          for s in sents:
            train_file.write(f'{s}\n')

          file_name = os.path.basename(train_file.name)

          msg.info(f'uploading {file_name}...')
          train_file.flush()
          api.upload_file(
            path_or_fileobj=train_file.name,
            path_in_repo=file_name,
            repo_type='dataset',
            repo_id=args.repo_id,
            token=args.hf_token
          )

          os.remove(train_file.name)

          sent_flush_iter += 1
          train_file = open(f'{args.assets_path}/brwac_train_{sent_flush_iter}.txt', 'a')
          msg.info(f'new flush idx: {idx}. file id: {sent_flush_iter}')
          sents = []

      if line.startswith('<s>'):
        sent_tokens = []
        continue

      if line.startswith('</s>'):
        sent = ' '.join(sent_tokens)
        sents.append(sent.lower())
        continue

      sent_tokens.append(line.replace('\n', ''))
