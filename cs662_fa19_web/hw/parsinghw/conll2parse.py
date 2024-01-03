#!/usr/bin/env python
# boilerplate code by Jon May (jonmay@isi.edu)
import argparse
import sys
import codecs
if sys.version_info[0] == 2:
  from itertools import izip
else:
  izip = zip
from collections import defaultdict as dd
import re
import os.path
import gzip
import tempfile
import shutil
import atexit
from logging import log, debug, warning, DEBUG
import logging
from enum import Enum
from parse import Word, projective, getLabel, applyLabel, stateToFeats

scriptdir = os.path.dirname(os.path.abspath(__file__))


reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')





def produceData(stack, _buffer, ofh, labeled=False, eager=False, fancy=False):
  """ recursively produce feature data given training data """
  if len(stack) == 0:
    return

  words, pos, labels = stateToFeats(stack, _buffer)

  label = getLabel(stack, _buffer, labeled, eager)
  if label is None:
    return
  stack, _buffer = applyLabel(stack, _buffer, label, labeled, eager)
  if fancy:
    pattern = "words=[{}]\tpos=[{}]\tlabels=[{}]\toutput={}\n"
  else:
    pattern = "{} {} {}\t{}\n"
  ofh.write(pattern.format(' '.join(words), ' '.join(pos), ' '.join(labels), label))
  produceData(stack, _buffer, ofh, labeled, eager, fancy)

def prepfile(fh, code):
  if type(fh) is str:
    fh = open(fh, code)
  ret = gzip.open(fh.name, code if code.endswith("t") else code+"t") if fh.name.endswith(".gz") else fh
  if sys.version_info[0] == 2:
    if code.startswith('r'):
      ret = reader(fh)
    elif code.startswith('w'):
      ret = writer(fh)
    else:
      sys.stderr.write("I didn't understand code "+code+"\n")
      sys.exit(1)
  return ret

def addonoffarg(parser, arg, dest=None, default=True, help="TODO"):
  ''' add the switches --arg and --no-arg that set parser.arg to true/false, respectively'''
  group = parser.add_mutually_exclusive_group()
  dest = arg if dest is None else dest
  group.add_argument('--%s' % arg, dest=dest, action='store_true', default=default, help=help)
  group.add_argument('--no-%s' % arg, dest=dest, action='store_false', default=default, help="See --%s" % arg)

def main():
    # https://www.aclweb.org/anthology/D14-1082

    # context is words, tags, labels
    # element of each is top 3 words on stack (s1,s2,s3), top 3 words on buffer(b1,b2,b3), s1lc1, s1rc1, s1lc2, s1rc2, s2lc1, s2rc1, s2lc2, s2rc2, s1lc1lc1, s1rc1rc1, s2lc1lc1, s2rc1rc1
    # tags for each of those 18, arcs for each of those except for the first 6.
  parser = argparse.ArgumentParser(description="given conll parse format, generate a variety of supervised labeled examples in tsv",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--infile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input file")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")
  parser.add_argument("--labeled", "-l", action='store_true', help="labeled arcs")
  parser.add_argument("--fancy", "-f", action='store_true', help="fancy output (used for debugging)")
  parser.add_argument("--eager", "-e", action='store_true', help="eager (instead of standard) arcs")



  try:
    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))

  workdir = tempfile.mkdtemp(prefix=os.path.basename(__file__), dir=os.getenv('TMPDIR', '/tmp'))

  def cleanwork():
    shutil.rmtree(workdir, ignore_errors=True)
  if args.debug:
    logging.basicConfig(level=DEBUG)
    debug(workdir)
  else:
    atexit.register(cleanwork)


  infile = prepfile(args.infile, 'r')
  outfile = prepfile(args.outfile, 'w')


  rootword = Word.getRoot()
  stack = [rootword]
  _buffer = []
  wordsByIndex = {0: rootword}
  for line in infile:
    # TODO: brittle
    if line.startswith("#"):
      continue
    elif len(line) == 1:
      if projective(_buffer, wordsByIndex):
        produceData(stack, _buffer, outfile, args.labeled, args.eager, args.fancy)
      rootword = Word.getRoot()
      stack = [rootword]
      _buffer = []
      wordsByIndex = {0: rootword}
    else:
      word = Word(line)
      _buffer.append(word)
      wordsByIndex[word.tokid] = word

if __name__ == '__main__':
  main()


# elsewhere i trained a model, it takes in the things produced here and learns them.
# now i need to:
# 1) given a sentence, create initial state (no problem, done here)
# 2) given a state, create a feature vector (no problem, done here)
# 3) encode a feature vector so the model can read it (done in learningstub)
# 4) get action decision (TODO; probably from learningstub)
# 5) apply decision to get new state, creating seeds for tree (TODO; here, maybe already done)
# 6) write heads and arcs when done (TODO)
