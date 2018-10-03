"""Demo server."""
import argparse
import bottle
import os
import shutil
import sys

import bna

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('out_dir', help='Temporary output directory.')
  parser.add_argument('--hostname', '-n', default='0.0.0.0', help='hostname.')
  parser.add_argument('--port', '-p', default=9000, type=int, help='port.')
  parser.add_argument('--debug', '-d', default=False, action='store_true', help='Debug mode')
  parser.add_argument('--train-dir', '-t', help='Path to trained parameters')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def run_model(paragraph, question):
  pred, beam = bna.query_server_single(paragraph, question)
  return beam[:10]

def main():
  if os.path.exists(OPTS.out_dir):
    shutil.rmtree(OPTS.out_dir)
  os.makedirs(OPTS.out_dir)
  print >> sys.stderr, 'Starting BiDAF-No-Answer server...'
  bna.start_server(OPTS.out_dir, verbose=True, train_dir=OPTS.train_dir)
  app = bottle.Bottle()

  @app.route('/')
  def index():
    return bottle.template('index')

  @app.route('/post_query', method='post')
  def post_query():
    paragraph = bottle.request.forms.getunicode('paragraph').strip()
    question = bottle.request.forms.getunicode('question').strip()
    beam = run_model(paragraph, question)
    return bottle.template('results', paragraph=paragraph, question=question, 
                           beam=beam)

  cur_dir = os.path.abspath(os.path.dirname(__file__))
  bottle.TEMPLATE_PATH.insert(0, os.path.join(cur_dir, 'views'))
  bottle.run(app, host=OPTS.hostname, port=OPTS.port, debug=OPTS.debug)


if __name__ == '__main__':
  OPTS = parse_args()
  main()
