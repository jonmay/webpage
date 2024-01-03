# central repository for common parse functions
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
import numpy as np
from enum import Enum

class Word():
  """ A word read off of conll dependency data format """
  def __init__(self, line=None, do_pos2=False):
    self.parent = None
    self.parentid = None
    self.label = None
    self.oldlabel="-"
    self.children = [] 
    if line is None:
      self.tokid = Word.ROOTID
      self.txt = self.normtxt = self.pos = "<root>"
    else:
      # brittle: 10 toks = from training, 6 toks = from test
      toks = line.strip().split('\t')
      if len(toks) >=6:
        tokid, txt, normtxt, pos1, pos2, posaddl = toks[:6]
        self.tokid = int(tokid)
        self.txt = txt
        self.normtxt = normtxt
        self.pos = pos2 if do_pos2 else pos1
        if len(toks) >=10:
          parent, label, hlabel, extra = toks[6:]
          self.parentid = int(parent)
          self.label = label
          self.oldlabel = self.label
      else:
        warn("Bad data: {}".format(line))
  ROOTID = 0
  def __repr__(self):
    if self.parent is not None:
      return "{}-{}({}) ^ #{}#".format(self.txt, self.pos, self.tokid, self.parent.tokid)
    else:
      return "{}-{}({}) ^ {}".format(self.txt, self.pos, self.tokid, self.parentid)
  def getRoot():
    return Word()
  def isRoot(word):
    return word.tokid == Word.ROOTID
  def setLabel(self, label):
    self.label = label
  def toCONLLString(self):
    ''' tab-separated tokid, txt, normtxt, pos, pos, -, parentid [as set], label [as set], -,  parentid.label [as read]'''
    output = []
    derid = self.parent.tokid if self.parent is not None else "<NULL>" # this shouldn't happen when parsing!
    for item in (self.tokid, self.txt, self.normtxt, self.pos, self.pos, None, derid, self.label, None, "{}.{}".format(self.parentid, self.oldlabel)):
      if item is None:
        output.append("_")
      else:
        output.append("{}".format(item))
    return '\t'.join(output)
  def addParent(self, parent):
    self.parent = parent
  def addLeftChild(self, child):
    self.children.insert(0, child)
  def addRightChild(self, child):
    self.children.append(child)
  def getDescendents(self, order, limit=None):
    """ get up to orderth left and right children. if they don't exist get None. """
    ret = []
    # yuck...
    metapool = [self.children, [x for x in reversed(self.children)]]
    assert(limit is None or limit=="left" or limit=="right")
    if limit=="left":
      metapool = [metapool[0],]
    elif limit=="right":
      metapool = [metapool[1],]
    for pool in metapool:
      for i in range(order):
        val = None
        if len(pool) > i:
          val = pool[i]
        ret.append(val)
    return ret

  def getCMDescendents(self):
    """ per chen & manning: get first and second left and right children (lc1, lc2, rc1, rc2) 
    and first left and right children of, respectively first left and right children (lc1lc1, rc1rc1)."""
    ret = self.getDescendents(2)
    gcparents = self.getDescendents(1) #lc1 and rc1
    if gcparents[0] is None:
      ret.extend([None,])
    else:
      ret.extend(gcparents[0].getDescendents(1, "left"))
    if gcparents[1] is None:
      ret.extend([None,])
    else:
      ret.extend(gcparents[1].getDescendents(1, "right"))
      
    return ret


class Actions(Enum):
  pass

class EagerActions(Enum):
  Right = 'right'
  Left = 'left'
  Shift = 'shift'
  Reduce = 'reduce'
  def isValid(self, stack, _buffer):
    # TODO!
    return True

class StandardActions(Enum):
  Right = 'right'
  Left = 'left'
  Shift = 'shift'
  def isValid(self, stack, _buffer):
    # can't be shift if buffer is empty
    if len(_buffer) == 0 and self is StandardActions.Shift:
      return False
    # can't be left or right if stack < 2
    if len(stack) < 2 and self is not StandardActions.Shift:
      return False
    # can't be left if root is second child
    if self is StandardActions.Left and Word.isRoot(stack[1]):
      return False
    return True
  
class OutputAction():
  def __init__(self, Actions, label=None):
    self.action = Actions
    self.label = label

  def __str__(self):
    if self.label is None:
      return self.action.value
    return "{}.{}".format(self.action.value, self.label)
  def isValid(self, stack, _buffer):
    return self.action.isValid(stack, _buffer)
def getAction(acstr, labeled, eager):
    toks = acstr.split('.')
    label = toks[1] if labeled else None
    action = toks[0]
    if eager:
        return OutputAction(EagerActions(action), label)
    else:
        return OutputAction(StandardActions(action), label)

def projective(words, wordsByIndex):
  """ given a list of words that have children/parents set, determine if the tree will be projective """
  for word in words:
    debug("Checking {}".format(word))
    wordid = word.tokid
    assert(wordid != Word.ROOTID)
    linkid = word.parentid
#    if linkid == Word.ROOTID:
#      continue
    # TODO: is there a nice low = (x < y ? x : y) in python?
    if wordid < linkid:
      lword = wordid
      gword = linkid
    else:
      lword = linkid
      gword = wordid
    for midid in range(lword+1, gword):
      mid = wordsByIndex[midid]
      midlink = mid.parentid
      # if midlink == Word.ROOTID:
      #   debug("{} ok with {} (root)".format(mid, word))
      #   continue
      if midlink < lword or midlink > gword:
        debug("{} NOT ok with {}".format(mid, word))
        warning("{} to {} contains nonprojective arc {}".format(wordsByIndex[lword], wordsByIndex[gword], mid))
        return False
      else:
        debug("{} ok with {}".format(mid, word))
  return True


def getLabel(stack, buffer, labeled=False, eager=False):
  """ given a stack and buffer with total information, produce the parsing operation """
  if eager:
    return getEagerLabel(stack, buffer, labeled)
  else:
    return getStandardLabel(stack, buffer, labeled)

def applyLabel(stack, buffer, label, labeled=False, eager=False):
  """ change stack and buffer based on label and return new items"""
  if eager:
    return applyEagerLabel(stack, buffer, label, labeled)
  else:
    return applyStandardLabel(stack, buffer, label, labeled)

  

def getStandardLabel(stack, buffer, labeled=False):
  """ determine the label and return """
  if len(stack) < 2:
    if len(buffer) < 1:
      debug("No stack. no buffer. No action")
      return None
    debug("Shifting at beginning")
    return OutputAction(StandardActions('shift'))
  debug("0: {} ({}). 1: {} ({})".format(stack[0].tokid, stack[0].parentid, stack[1].tokid, stack[1].parentid))
  # left if possible: top is parent and next is direct child (will remove the child and put back the parent)
  if stack[0].tokid == stack[1].parentid:
    label = stack[1].label if labeled else None
    debug("labeling left")
    return OutputAction(StandardActions('left'), label)
  # right if next is parent and top is child and no other children of the child on the buffer (will remove the child)
  elif stack[1].tokid == stack[0].parentid:
    debug("candidate for right")
    seen = False
    for item in buffer:
      if item.parentid == stack[0].tokid:
        seen = True
        break
    if not seen:
      debug("labeling right")
      label = stack[0].label if labeled else None
      return OutputAction(StandardActions('right'), label)
    else:
      return OutputAction(StandardActions('shift'))
  else:
    debug("no relation. shift")
    return OutputAction(StandardActions('shift'))


def applyStandardLabel(stack, buffer, label, labeled=False):
  """ change stack and buffer based on label and return new items """
  # if actually parsing we'd insert parent and label info
  if label.action == StandardActions.Left:
    #assert(stack[0].tokid==stack[1].parentid and (not labeled or label.label == stack[1].label))
    parent = stack.pop(0)
    child = stack.pop(0)
    assert(child.parent is None)
    child.parent = parent
    if labeled:
      child.setLabel(label.label)

    parent.addLeftChild(child)
    stack.insert(0, parent)
  elif label.action == StandardActions.Right:
    #assert(stack[1].tokid==stack[0].parentid and (not labeled or label.label == stack[0].label))
    child = stack.pop(0)
    parent = stack[0]
    assert(child.parent is None)
    child.parent = parent
    if labeled:
      child.setLabel(label.label)

    parent.addRightChild(child)
  else:
    assert(len(buffer) > 0)
    stack.insert(0, buffer.pop(0))
  return stack, buffer

def applyEagerLabel(stack, buffer, label, labeled=False):
  """ change stack and buffer based on label and return new items """
  # if actually parsing we'd insert parent and label info
  if label.action == EagerActions.Left:
    assert(stack[0].parentid==buffer[0].tokid and (not labeled or label.label == stack[0].label))
    child = stack.pop(0)
    parent = buffer[0]
    assert(child.parent is None)
    child.parent = parent
    parent.addLeftChild(child)
  elif label.action == EagerActions.Right:
    assert(stack[0].tokid==buffer[0].parentid and (not labeled or label.label == buffer[0].label))
    child = buffer.pop(0)
    parent = stack[0]
    assert(child.parent is None)
    child.parent = parent
    parent.addRightChild(child)
    stack.insert(0, child)
  elif label.action == EagerActions.Shift:
    assert(len(buffer) > 0)
    stack.insert(0, buffer.pop(0))
  return stack, buffer


def getEagerLabel(stack, buffer, labeled=False):
  """ determine the label and return """
  if len(stack) < 1 and len(buffer) < 1:
    return None
  # left if possible: buf top is parent and stack top is direct child (will remove the child)
  if stack[0].parent == buffer[0].tokid:
    label = stack[0].label if labeled else None
    return OutputAction(EagerActions('left'), label)
  # reduce right if stack top is parent and buf top is direct child and no other children of the child on the buffer (will move the buffer over)
  elif stack[0].tokid == buffer[0].parent:
    label = buffer[0].label if labeled else None
    return OutputAction(EagerActions('right'), label)
  else:
    # reduce if top of stack has no more children
    seen = False
    for item in buffer:
      if item.parent == stack[0].tokid:
        seen = True
        break
    if not seen:
      return OutputAction(EagerActions('reduce'))
    else:
      return OutputAction(EagerActions('shift'))

def getModelFeats(input, mapper):
  return ([x for x in map(lambda x: mapper[x], input)])

def stateToFeats(stack, _buffer):
  """ turn state (stack and buffer) into feature space fed to model """
  stackwindow = (stack+[None, None, None])[:3]
  buffwindow = (_buffer+[None, None, None])[:3]
  descparents = (stack+[None, None])[:2]
  descwindow = []
  debug("stack: {}; window: ({}) {}".format(stack, len(stackwindow), stackwindow))
  debug("buffer: {}; window: ({}) {}".format(_buffer, len(buffwindow), buffwindow))
  
  for parent in descparents:
    if parent is None:
      descwindow.extend([None]*6)
    else:
      descwindow.extend(parent.getCMDescendents())
  words = [("None" if x is None else x.normtxt) for x in stackwindow+buffwindow+descwindow]
  pos = [("None" if x is None else x.pos) for x in stackwindow+buffwindow+descwindow]
  labels = [("None" if x is None else x.label) for x in descwindow]
  return words, pos, labels
