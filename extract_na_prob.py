#!/usr/bin/env python3
"""Extract NA probs from BiDAF-no-answer model."""
import argparse
import gzip
import json
import pickle
import sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('')
  parser.add_argument('eval_file', metavar='eval.pkl.gz')
  parser.add_argument('data_file', metavar='data_single.json')
  parser.add_argument('out_file', metavar='na_prob.json')
  parser.add_argument('--argmax', '-a', action='store_true',
                      help='Check if NA is argmax prediction')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def main():
  with gzip.open(OPTS.eval_file) as f:
    e = pickle.load(f)
  with open(OPTS.data_file) as f:
    d = json.load(f)
  qids = d['ids']
  na_probs = e['na']
  out_obj = {}
  if OPTS.argmax:
    y1_list = e['yp']
    y2_list = e['yp2']
    for qid, na_prob, y1_list, y2_list in zip(qids, na_probs, y1_list, y2_list):
      best_ans_prob = max(y1_list[0]) * max(y2_list[0])
      if best_ans_prob > na_prob:
        out_obj[qid] = 0.0
      else:
        out_obj[qid] = 1.0
  else:
    for qid, na_prob in zip(qids, na_probs):
      out_obj[qid] = na_prob
  with open(OPTS.out_file, 'w') as f:
    json.dump(out_obj, f)

if __name__ == '__main__':
  OPTS = parse_args()
  main()

