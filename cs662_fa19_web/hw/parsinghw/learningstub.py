from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras import utils
import pickle
import numpy as np
fh = open('en_ewt-ud-dev.conllu.ul.dep.txt')
classes = {}
classes = set()
words = set()
word2index = {}
index2word = {}
word2class = {}
class2word = {}
for line in fh:
  toks, label = line.strip().split('\t')
  toks = toks.split()
  for tok in toks:
    words.add(tok)
  classes.add(label)

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
# TODO: adagrad with 0.5 dropout

index2class = {}
class2index = {}
for idx, tok in enumerate(classes):
  class2index[tok] = idx
  index2class[idx] = tok
train_data = []
train_labels = []
label_size = len(class2index.keys())
index = {'w2i':word2index, 'i2w':index2word, 'c2i':class2index, 'i2c':index2class}
pickle.dump(index, open('index.pickle', 'wb'))

fh = open('en_ewt-ud-dev.conllu.ul.dep.txt')
for line in fh:
  toks, label = line.strip().split('\t')
  toks = toks.split()
  tokids = []
  for tok in toks:
    tokids.append(word2index[tok])
  train_data.append(tokids)
  train_labels.append(class2index[label])

#?
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 50, input_length=48))# inputs are 48x50 = 2400
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(200, activation=tf.nn.tanh))# matrix is 2400*200 so far
model.add(keras.layers.Dense(label_size, activation=tf.nn.softmax))# matrix is 200x3; vector is softmaxed 3

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
p_x_t = train_data[:40000]
p_y_t = train_labels[:40000]
x_val = train_data[40000:]
y_val = train_labels[40000:]
      

p_x_t = np.array(p_x_t)
p_y_t = utils.to_categorical(np.array(p_y_t), label_size)
x_val = np.array(x_val)
y_val = utils.to_categorical(np.array(y_val), label_size)
print("train x {} train y {} dev x {} dev y {}".format(p_x_t.shape, p_y_t.shape, x_val.shape, y_val.shape))
print(p_x_t[0])
print(p_y_t[0])
history = model.fit(p_x_t, p_y_t, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

predictions = model.predict(x_val)

model.save('model.h5')

new_model = keras.models.load_model('model.h5')

new_predictions = new_model.predict(x_val)

print([index2class[x] for x in [np.argmax(y) for y in predictions[:5]]])
print([index2class[x] for x in [np.argmax(y) for y in new_predictions[:5]]])

np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

# TODO: need to save off the vocabulary mapping code too


# now actually decode the dev (or test) and output steps
# live decoding requires dynamic-time recalculation of features

# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/


