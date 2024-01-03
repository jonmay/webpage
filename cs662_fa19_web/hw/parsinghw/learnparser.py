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
from tensorflow import keras
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.python.keras import utils
from parse import getModelFeats
import pickle
import numpy as np

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



def main():
  parser = argparse.ArgumentParser(description="REPLACE WITH DESCRIPTION",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--trainfile", "-t", nargs='?', required=True, help="input training file")
  parser.add_argument("--devfile", "-d", nargs='?', required=True, help="input dev file")
  parser.add_argument("--epochs", "-e", type=int, default=40, help="epochs")
  parser.add_argument("--outfile", "-o", nargs='?', required=True, help="output model file")
  parser.add_argument("--vocabfile", "-v", nargs='?', type=argparse.FileType('wb'), required=True, help="output vocabulary file")




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


  trainfile = open(args.trainfile, 'r')
  vocabfile = prepfile(args.vocabfile, 'w')

  classes = set()
  words = set()

  for line in trainfile:
    toks, label = line.strip().split('\t')
    toks = toks.split()
    for tok in toks:
      words.add(tok)
    classes.add(label)
  word2index = {}
  index2word = {}

  for idx, tok in enumerate(words, start=4):
    word2index[tok] = idx
    index2word[idx] = tok

  word2index["<PAD>"] = 0
  word2index["<START>"] = 1
  word2index["<UNK>"] = 2
  word2index["<UNUSED>"] = 3
  model = keras.Sequential()
  vocab_size=len(word2index.keys())

  # TODO: initialize random to -0.01, 0.01
  # TODO: initialize words to collobert
  # TODO: l2 regularization


  index2class = {}
  class2index = {}
  for idx, tok in enumerate(classes):
    class2index[tok] = idx
    index2class[idx] = tok
  train_data = []
  train_labels = []
  dev_data = []
  dev_labels = []
  label_size = len(class2index.keys())
  index = {'w2i':word2index, 'i2w':index2word, 'c2i':class2index, 'i2c':index2class}
  pickle.dump(index, args.vocabfile)

  # to account for OOV
  mapper = dd(lambda: word2index["<UNK>"])
  mapper.update(word2index)
  trainfile = open(args.trainfile, 'r')
  for line in trainfile:
    toks, label = line.strip().split('\t')
    toks = toks.split()
    train_data.append(getModelFeats(toks, mapper))
    train_labels.append(class2index[label])
  devfile = open(args.devfile, 'r')
  for line in devfile:
    toks, label = line.strip().split('\t')
    toks = toks.split()
    dev_data.append(getModelFeats(toks, mapper))
    dev_labels.append(class2index[label])
  train_data = np.array(train_data)
  dev_data = np.array(dev_data)
  train_labels = utils.to_categorical(np.array(train_labels), label_size)
  dev_labels = utils.to_categorical(np.array(dev_labels), label_size)
  
  model = keras.Sequential()
  model.add(keras.layers.Embedding(vocab_size, 50, input_length=48, activity_regularizer=l2(0.01)))# inputs are 48x50 = 2400
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(200, activation=tf.nn.tanh, activity_regularizer=l2(0.01)))# matrix is 2400*200 so far
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Dense(label_size, activation=tf.nn.softmax, activity_regularizer=l2(0.01)))# matrix is 200x3; vector is softmaxed 3

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
  print(model.summary())
  
  print("train x {} train y {} dev x {} dev y {}".format(train_data.shape, train_labels.shape, dev_data.shape, dev_labels.shape))
  print(train_data[0])
  print(train_labels[0])
  history = model.fit(train_data, train_labels, epochs=args.epochs, batch_size=512, validation_data=(dev_data, dev_labels), verbose=1)
  
  predictions = model.predict(dev_data)
  model.save(args.outfile)

if __name__ == '__main__':
  main()
