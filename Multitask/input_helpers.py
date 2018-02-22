import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
#from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip
from random import random
from preprocess import MyVocabularyProcessor
import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

class InputHelper(object):
    pre_emb = dict()
    vocab_processor = None
    def cleanText(self, s):
        s = re.sub(r"[^\x00-\x7F]+"," ", s)
        s = re.sub(r'[\~\!\`\^\*\{\}\[\]\#\<\>\?\+\=\-\_\(\)]+',"",s)
        s = re.sub(r'( [0-9,\.]+)',r"\1 ", s)
        s = re.sub(r'\$'," $ ", s)
        s = re.sub('[ ]+',' ', s)
        return s.lower()

    def getVocab(self,vocab_path, max_document_length,filter_h_pad):
        if self.vocab_processor==None:
            print('locading vocab')
            vocab_processor = MyVocabularyProcessor(max_document_length-filter_h_pad,min_frequency=0)
            self.vocab_processor = vocab_processor.restore(vocab_path)
        return self.vocab_processor

    def loadW2V(self, emb_path, type="bin"):
        print("Loading W2V data...")
        num_keys = 0
        if type=="textgz":
            # this seems faster than gensim non-binary load
            for line in gzip.open(emb_path):
                l = line.strip().split()
                st=l[0].lower()
                self.pre_emb[st]=np.asarray(l[1:])
            num_keys=len(self.pre_emb)
        if type=="text":
            # this seems faster than gensim non-binary load
            for line in open(emb_path):
                l = line.strip().split()
                st=l[0].lower()
                self.pre_emb[st]=np.asarray(l[1:])
            num_keys=len(self.pre_emb)
        else:
            self.pre_emb = Word2Vec.load_word2vec_format(emb_path,binary=True)
            self.pre_emb.init_sims(replace=True)
            num_keys=len(self.pre_emb.vocab)
        print("loaded word2vec len ", num_keys)
        gc.collect()


    def loadW2VFloat(self, emb_path, type="text"):
        #print("Loading W2V data...")
        num_keys = 0
        if type=="textgz":
            # this seems faster than gensim non-binary load
            for line in gzip.open(emb_path):
                l = line.strip().split()
                st=l[0].lower()
                self.pre_emb[st]=np.asarray(l[1:], dtype=np.float32)
            num_keys=len(self.pre_emb)
        if type=="text":
            # this seems faster than gensim non-binary load
            for line in open(emb_path):
                l = line.strip().split()
                st,emb=l[0].lower(),[]
                for val in l[1:]:
                    try:
                        v = float(val)
                        emb.append(v)
                    except:
                        emb.append(0)
                self.pre_emb[st]=np.asarray(emb)
            num_keys=len(self.pre_emb)
        else:
            self.pre_emb = Word2Vec.load_word2vec_format(emb_path,binary=True)
            self.pre_emb.init_sims(replace=True)
            num_keys=len(self.pre_emb.vocab)
        #print("loaded word2vec len ", num_keys)
        gc.collect()

    def mergeW2V(self, emb_path, type="bin"):
        print("Loading W2V data...")
        num_keys = 0
        if type == "textgz":
            # this seems faster than gensim non-binary load
            for line in gzip.open(emb_path):
                l = line.strip().split()
                st = l[0].lower()
                self.pre_emb[st] = np.asarray(l[1:])
            num_keys = len(self.pre_emb)
        if type == "text":
            # this seems faster than gensim non-binary load
            i = 0
            for line in open(emb_path):
                l = line.strip().split()
                st = l[0].lower()
                if st in self.pre_emb:
                    continue
                else:
                    i += 1
                    #if i % 10000 == 0:
                    #    print st
                    self.pre_emb[st] = np.asarray(l[1:])
            num_keys = len(self.pre_emb)
            print (num_keys)
        else:
            self.pre_emb = Word2Vec.load_word2vec_format(emb_path, binary=True)
            self.pre_emb.init_sims(replace=True)
            num_keys = len(self.pre_emb.vocab)
        print("loaded word2vec len ", num_keys)
        gc.collect()

    def deletePreEmb(self):
        self.pre_emb=dict()
        gc.collect()

    def getTsvData(self, filepath):
        print("Loading training data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        # positive samples from file
        for line in open(filepath):
            l=line.strip().split("\t")
            if len(l)<2:
                continue
            if random() > 0.5:
                x1.append(l[0].lower())
                x2.append(l[1].lower())
            else:
                x1.append(l[1].lower())
                x2.append(l[0].lower())
            y.append(int(l[2]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)

    def getTsvDataCharBased(self, filepath):
        print("Loading training data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        # positive samples from file
        for line in open(filepath):
            l=line.strip().split("\t")
            if len(l)<2:
                continue
            if random() > 0.5:
               x1.append(l[0].lower())
               x2.append(l[1].lower())
            else:
               x1.append(l[1].lower())
               x2.append(l[0].lower())
            y.append(1)#np.array([0,1]))
        # generate random negative samples
        combined = np.asarray(x1+x2)
        shuffle_indices = np.random.permutation(np.arange(len(combined)))
        combined_shuff = combined[shuffle_indices]
        for i in xrange(len(combined)):
            x1.append(combined[i])
            x2.append(combined_shuff[i])
            y.append(0) #np.array([1,0]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)


    def getTsvTestData(self, filepath):
        print("Loading testing/labelled data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        # positive samples from file
        for line in open(filepath):
            #print line
            l=line.strip().split("\t")
            if len(l)<3:
                continue
            x1.append(l[1].lower())
            x2.append(l[2].lower())
            y.append(int(l[0])) #np.array([0,1]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)

    def getAquaintData(self, filepath):
        print("Loading testing/labelled data from " + filepath)
        #x0 = []
        #x1 = []
        #x2 = []
        x3 = []
        #x4 = []
        x5 = []
        x6 = []
        # positive samples from file
        for line in open(filepath):

            l = line.strip().split("\t")
            #if len(l)<3:
            #print line
            #print l
            if len(l) < 5:
                print ('error in data')
                continue
            #x0.append(l[0].lower())
            #x1.append(l[1].lower())
            #x2.append(l[2].lower())
            x3.append(l[3].lower())
            #x4.append(l[4].lower())
            x5.append(l[5].lower())
            x6.append(l[6].lower())
        return np.asarray(x3), np.asarray(x5), np.asarray(x6)



    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        print(data)
        print(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
                
    def dumpValidation(self,x1_text,x2_text,y,shuffled_index,dev_idx,i):
        print("dumping validation "+str(i))
        x1_shuffled=x1_text[shuffled_index]
        x2_shuffled=x2_text[shuffled_index]
        y_shuffled=y[shuffled_index]
        x1_dev=x1_shuffled[dev_idx:]
        x2_dev=x2_shuffled[dev_idx:]
        y_dev=y_shuffled[dev_idx:]
        del x1_shuffled
        del y_shuffled
        with open('validation.txt'+str(i),'w') as f:
            for text1,text2,label in zip(x1_dev,x2_dev,y_dev):
                f.write(str(label)+"\t"+text1+"\t"+text2+"\n")
            f.close()
        del x1_dev
        del y_dev
    
    # Data Preparatopn
    # ==================================================
    
    
    def getDataSets(self, training_paths, max_document_length, percent_dev, batch_size, is_char_based):
        if is_char_based:
            x1_text, x2_text, y=self.getTsvDataCharBased(training_paths)
        else:
            x1_text, x2_text, y=self.getTsvData(training_paths)
        # Build vocabulary
        x1_large_text, x2_large_text, y_large_text = self.getAquaintData("aquaint.tsv")
        print("Building vocabulary")
        vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=0,is_char_based=is_char_based)
        #vocab_processor.fit_transform(np.concatenate((x2_text,x1_text),axis=0))
        vocab_processor.fit_transform(np.concatenate((x2_text, x1_text, x2_large_text, x1_large_text), axis=0))
        print("Length of loaded vocabulary ={}".format( len(vocab_processor.vocabulary_)))
        i1=0
        train_set=[]
        dev_set=[]
        sum_no_of_batches = 0
        x1 = np.asarray(list(vocab_processor.transform(x1_text)))
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))
        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x1_shuffled = x1[shuffle_indices]
        x2_shuffled = x2[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1*len(y_shuffled)*percent_dev//100
        del x1
        del x2
        # Split train/test set
        self.dumpValidation(x1_text,x2_text,y,shuffle_indices,dev_idx,0)
        # TODO: This is very crude, should use cross-validation
        x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
        x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))
        sum_no_of_batches = sum_no_of_batches+(len(y_train)//batch_size)
        train_set=(x1_train,x2_train,y_train)
        dev_set=(x1_dev,x2_dev,y_dev)
        gc.collect()
        return train_set,dev_set,vocab_processor,sum_no_of_batches

    #This function is taking vocabulary file data and converting it into integers
    #We can convert these integer indices to word embedding by using the W matrix and tensorflow.nn.lookup function
    def getTestDataSet(self, data_path, vocab_path, max_document_length):
        x1_temp,x2_temp,y = self.getTsvTestData(data_path)
        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)
        print (len(vocab_processor.vocabulary_))
        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1,x2, y


    def getAquaintTestDataSet(self, data_path, vocab_path, max_document_length):
        x1_temp,x2_temp,y = self.getAquaintData(data_path)
        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)
        print (len(vocab_processor.vocabulary_))
        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1,x2, y
