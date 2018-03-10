import re
import numpy as np
import gc
from gensim.models.word2vec import Word2Vec
import gzip
from preprocess import MyVocabularyProcessor

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

    #this loads any word2vec file
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
            i=0
            for line in open(emb_path):
                l = line.strip().split()
                st,emb=l[0].lower(),[]
                if st in self.pre_emb:
                    continue
                else:
                    i += 1
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

    def deletePreEmb(self):
        self.pre_emb=dict()
        gc.collect()


