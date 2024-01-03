#!/usr/bin/env python3
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
import numpy as np
from absl.logging import log, debug, warning, DEBUG
import absl.logging as logging


scriptdir = os.path.dirname(os.path.abspath(__file__))


reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


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

def softmax(x):
  return np.exp(x)/np.sum(np.exp(x), axis=1)

def tanhderiv(x):
  ''' or 1 - (((np.exp(x)-np.exp(-x))**2)/((np.exp(x)+np.exp(-x))**2)) '''
  return 1 - (np.tanh(x)**2)


def onehot(shape, values):
  ''' make a one-hot of the given values in the given shape '''
  assert(min(values) >= 0)
  assert(max(values) < shape[1])
  return np.eye(shape[1])[np.array([values]).reshape(-1)]

def rebatch_data(train_data, batch_size):
  ''' make random batches out of train data '''
  # permute train data
  # slice every batch_size elements, 

def train_model(train_data, dev_data, data_size, epochs, batch_size, vocab_size, embedding_size, hidden_size, output_size):
  ''' train one layer feed forward fully connected network. see charniak ch. 1 (but also embeddings) '''

  # initialize
  embeddings = np.matrix(np.random.uniform(low=-0.01, high=0.01, size=(vocab_size, embedding_size))) # todo: load these externally?
  hW = np.matrix(np.random.uniform(low=-0.01, high=0.01, size=(data_size*embedding_size, hidden_size)))
  hB = np.matrix(np.random.uniform(low=-0.01, high=0.01, size=(1, hidden_size)))
  oW = np.matrix(np.random.uniform(low=-0.01, high=0.01, size=(hidden_size, output_size)))
  oB = np.matrix(np.random.uniform(low=-0.01, high=0.01, size=(1, output_size)))

  for epoch in range(epochs):
    # form data into random batches
    proc_train = rebatch_data(train_data, batch_size)
    for feats, labels in proc_train: # feats is (batch_size, data_size). labels is (1, batch_size)
      # look up embeddings and concatenate. Hint: use advanced slicing and reshape!
      assert(feats.shape[1]) == data_size)
      input = embeddings[feats,].reshape(-1, 1, data_size*embedding_size) # fu
      # pass through hidden layer # TODO: change to relu?
      prehidden = input*hW+hB
      throughHidden = np.maximum(prehidden, 0) # relu
      # pass through hidden-to-output layer, including bias
      throughOutput = softmax((throughHidden*oW)+oB)

      # logitgradient: partial derivative of the loss relative to the logit (just the probability part)      
      # turn truthprob into (batchsize, output_size); replicate to output_size and mask (elementwise multiply) with a one-hot

      mask = onehot((batch_size, output_size), labels)
      logitgradient = mask - throughOutput # 

      
      #logitgradient = np.multiply(np.repeat(truthprob-1, output_size).reshape(batch_size, output_size), mask) # batchxoutput
      # truthprob, where value is not truth
      # THIS IS WRONG
      # logitgradient = logitgradient + (np.multiply(np.repeat(truthprob, output_size).reshape(batch_size, output_size), 1-mask))

      doB = logitgradient
      doW = np.matmul(throughHidden.transpose(), logitgradient)

      # partial w/r/t relu:
      relugradient = np.where(prehidden>0, 1, 0)
      # not sure if i'm supposed to multiply both by 
      dhB = relugradient
      dhW = np.matmul(np.matmul(prehidden, relugradient), 

                      
      oB -= doB
      oW -= doW
      hW -= dhW
                      
      # TODO: print loss change
      # TODO: mask out each update; should not get worse
      
      # derivative of logit relative to output bias is ones.
      # derivative of logit relative to output weights is "throughHidden"
      # update for ob is loss_relative_logit*logit_relative_ob = lossgradient
      # update for oW is loss_relative_logit*logit_relative_ow = lossgradient*throughHidden
      
      # eisenstein terminology: l = -e_y * log(y) (e_y are one-hot of truth; this is cross entropy)
      #                         y = softmax(theta_zy*z + b)
      #                         z = f(theta_xz*x)
      #  so dl/dx = dl/dy * dy/dz * dz/dx
      #     dl/dz = dl/dy * dy/dz


      # dl/dz = truthprob, or truthprob-1 for the cells aligned with the truth * the "throughOutput"

      # dl/dx = dl/dz (just calculated) * dz/dx = tanhderiv(x)
      # charniak defines things a bit differently

      
      # derivative of loss relative to throughHidden = 
      # calculate loss: negative log prob of the truth

      # calculate gradient for oW and oB; update
      # calculate gradient for hW; update
      # dhW = np.multiply(logitgradient, tanhderiv(hW)) # wrong?
      # hW local gradient: let throughHidden be x. Then gradient for local is 1-tanh^2(x) = 1 - (e^x - e^(-x))^2/(e^x + e^(-x))^2
      # calculate gradient for embeddings; update
      

  
def main():
    parser = argparse.ArgumentParser(description="REPLACE WITH DESCRIPTION",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--infile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input file")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")




  try:
    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))

  workdir = tempfile.mkdtemp(prefix=os.path.basename(__file__), dir=os.getenv('TMPDIR', '/tmp'))

  def cleanwork():
    shutil.rmtree(workdir, ignore_errors=True)
  if args.debug:
    print(workdir)
  else:
    atexit.register(cleanwork)


  infile = prepfile(args.infile, 'r')
  outfile = prepfile(args.outfile, 'w')


  for line in infile:
    outfile.write(line)

if __name__ == '__main__':
  main()
