"""Running the BiDAF-No-Answer model."""
import argparse
import atexit
import collections
import errno
import json
import os
import requests
import shutil
import socket
import subprocess
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEAM_SIZE = 10  # Beam size when computing approximate expected F1 score
DEVNULL = open(os.devnull, 'w')
OPTS = None

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 6070
SERVER_URL = 'http://%s:%d/query' % (SERVER_HOST, SERVER_PORT)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('model', help='Name of model', choices=['debug-server'])
  parser.add_argument('filename', help='SQuAD JSON data.')
  parser.add_argument('out_dir', help='Temporary output directory.')
  parser.add_argument('--train-dir', '-t', help='Path to trained parameters')
  #parser.add_argument('--pred-file', '-p', help='Write preds to this file')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def get_phrase(context, words, span):
  """Reimplementation of bidaf_no_answer.squad.utils.get_phrase."""
  start, stop = span
  char_idx = 0
  char_start, char_stop = None, None
  for word_idx, word in enumerate(words):
    char_idx = context.find(word, char_idx)
    if word_idx == start:
      char_start = char_idx
    char_idx += len(word)
    if word_idx == stop - 1:
      char_stop = char_idx
  return context[char_start:char_stop]

def get_y_pred_beam(start_probs, end_probs, context, words, beam_size=BEAM_SIZE,
                    na_prob=None):
  beam = []
  for i, p_start in enumerate(start_probs):
    for j, p_end in enumerate(end_probs):
      if i <= j:
        phrase = get_phrase(context, words, (i, j+1))
        beam.append((phrase, i, j + 1, p_start * p_end))
  if na_prob is not None:
    beam.append(('', -1, -1, na_prob))
  beam.sort(key=lambda x: x[3], reverse=True)
  return beam[:beam_size]

def extract_fwd_files(out_dir):
  """Extract information from running the model forward."""
  # Run files
  data_path = os.path.join(out_dir, 'data', 'data_single.json')
  shared_path = os.path.join(out_dir, 'data', 'shared_single.json')
  eval_pklgz_path = os.path.join(out_dir, 'preds', 'eval.pklgz')

  # Due to python2/3 incompatibilities, use python3 to convert pklgz to JSON
  eval_json_path = os.path.join(out_dir, 'preds', 'eval.json')
  subprocess.check_call([
      'python3', '-c', 
      'import gzip, json, pickle; json.dump(pickle.load(gzip.open("%s")), open("%s", "w", encoding="utf-8"))' % (eval_pklgz_path, eval_json_path)])

  # Extract probability scores
  with open(data_path) as f:
    data_single_obj = json.load(f)
  with open(eval_json_path) as f:
    eval_obj = json.load(f)
  with open(shared_path) as f:
    shared_single_obj = json.load(f)

  return data_single_obj, eval_obj, shared_single_obj

def start_server(out_dir, verbose=False, nlp_cluster=False, train_dir=None):
  """Start BNA server, return the process once it's up.

  Args:
    out_dir: Directory for scratch work.
    verbose: If True, print subprocess output.
    nlp_cluster: If True, configure python3 to work on NLP cluster
    train_dir: If provided, use params from this training run
  """
  if verbose:
    pipeout = None
  else:
    pipeout = DEVNULL
  inter_dir = os.path.join(out_dir, 'inter_single')
  os.mkdir(inter_dir)
  env = os.environ.copy()
  env['PYTHONPATH'] = ROOT_DIR
  eval_pklgz_path = os.path.join(inter_dir, 'eval.pklgz')
  eval_json_path = os.path.join(inter_dir, 'eval.json')
  data_path = os.path.join(inter_dir, 'data_single.json')
  shared_path = os.path.join(inter_dir, 'shared_single.json')
  target_path = os.path.join(out_dir, 'preds.json')
  if nlp_cluster:
    env['LD_LIBRARY_PATH'] = 'libc/lib/x86_64-linux-gnu/:libc/usr/lib64/'
    python3_args = [
        'libc/lib/x86_64-linux-gnu/ld-2.17.so', 
        '/u/nlp/packages/anaconda2/envs/robinjia-py3/bin/python',
    ]
  else:
    python3_args = ['python3']
  if train_dir:
    save_args = [
        '--out_base_dir', train_dir,
        '--shared_path', os.path.join(train_dir, 'basic/00/shared.json'),
    ]
  else:
    raise NotImplementedError
    save_args = [
        '--load_path', get_load_path(MODEL_NUM),
        '--shared_path', get_shared_path(MODEL_NUM),
    ]
  run_args = python3_args + [
      '-O', '-m', 'basic.cli', '--data_dir', inter_dir, 
      '--eval_path', eval_pklgz_path, '--nodump_answer', 
      '--eval_num_batches', '0', '--mode', 'server', '--batch_size', '1',
      '--len_opt', '--cluster', '--cpu_opt'] + save_args
      # python3 -O disables assertions
  if verbose:
    print >> sys.stderr, run_args
  p = subprocess.Popen(run_args, env=env, stdout=pipeout, stderr=pipeout)
  atexit.register(p.terminate)

  # Keep trying to connect until the server is up
  s = socket.socket()
  while True:
    time.sleep(1)
    try:
      s.connect((SERVER_HOST, SERVER_PORT))
    except socket.error as e:
      if e.errno != errno.ECONNREFUSED:
        # Something other than Connection refused means server is running
        break
  s.close()
  return p

def query_server_raw(dataset, verbose=False):
  response = requests.post(SERVER_URL, json=dataset)
  response_json = json.loads(response.text)
  data_single_obj = response_json['data_single']
  eval_obj = response_json['eval']
  shared_single_obj = response_json['shared_single']
  pred_obj = response_json['predictions']
  return data_single_obj, eval_obj, shared_single_obj, pred_obj

def query_server_squad(dataset, verbose=False, return_eval=False):
  """Query server using SQuAD-formatted data."""
  data_single_obj, eval_obj, shared_single_obj, pred_obj = query_server_raw(
      dataset, verbose=verbose)
  json_ids = data_single_obj['ids']
  id2beam = {}  # Beam of possible non-empty answers
  id2na = {}  # Probability of predicting no answer
  for i, cur_id in enumerate(json_ids):
    start_probs = eval_obj['yp'][i][0]
    end_probs = eval_obj['yp2'][i][0]
    a_idx, p_idx = data_single_obj['*x'][i]
    context = shared_single_obj['p'][a_idx][p_idx]
    words = shared_single_obj['x'][a_idx][p_idx][0]
    answers = data_single_obj['answerss'][i]
    na_prob = eval_obj['na'][i]
    raw_beam = get_y_pred_beam(start_probs, end_probs, context, words,
                               na_prob=na_prob)
    phrase_beam = []
    for phrase, start, end, prob in raw_beam:
      phrase_beam.append((phrase, prob))
    id2na[cur_id] = na_prob
    id2beam[cur_id] = phrase_beam
  response = (pred_obj, id2beam)
  if return_eval:
    response += (eval_obj,)
  return response

def query_server_single(paragraph, question, **kwargs):
  data = {
      'version': 'v1.1',
      'data': [{
          'title': '',
          'paragraphs': [{
              'context': paragraph,
              'qas': [{
                  'question': question,
                  'id': 'single',
                  'answers': []
              }]
          }]
      }]
  }
  response = query_server_squad(data, **kwargs)
  pred = response[0]['single']
  beam = response[1]['single']
  return (pred, beam) + response[2:]

def debug_server(json_filename, out_dir, verbose=False, **kwargs):
  t0 = time.time()
  process = start_server(out_dir, verbose=verbose, **kwargs)
  t1 = time.time()
  if verbose:
    print >> sys.stderr, 'Server startup took %.2f seconds' %  (t1 - t0)
  with open(json_filename) as f:
    dataset = json.load(f)
  response = query_server_squad(dataset, verbose=verbose, return_eval=True)
  if verbose:
    print response
  t2 = time.time()
  if verbose:
    print >> sys.stderr, 'Query took %.2f seconds' %  (t2 - t1)
  return response

def main():
  if os.path.exists(OPTS.out_dir):
    shutil.rmtree(OPTS.out_dir)
  os.makedirs(OPTS.out_dir)
  if OPTS.model == 'debug-server':
    debug_server(OPTS.filename, OPTS.out_dir, verbose=True,
                 train_dir=OPTS.train_dir)
  else:
    raise ValueError('Unrecognized model "%s"' % OPTS.model)

if __name__ == '__main__':
  OPTS = parse_args()
  main()
