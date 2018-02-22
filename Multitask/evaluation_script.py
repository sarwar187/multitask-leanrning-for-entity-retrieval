from __future__ import division
from __future__ import print_function
import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import sys
from input_helpers import InputHelper

def generate_random_score(filepath):
    #print("Generating random score " + filepath)
    f_out = open(filepath + '.scored', 'w+')
    for line in open(filepath):
        line = line.strip() + "\t" + str(np.random.uniform())
        #print(line, file=f_out)
        #print >> f_out, line
    return filepath + '.scored'

def print_score(filepath, score, script_id):
    #print("Printing Score " + filepath)
    filepathfinal = filepath + script_id + ".scored"
    f_out = open(filepathfinal, 'w+')
    i=0
    sharefile = open(filepath,'r')
    for line in sharefile:
        line = line.strip() + "\t" + str(score[i])
        i+=1
        print(line, file=f_out)
        #print >> f_out, line
    f_out.close()
    sharefile.close()
    return filepathfinal

#get the testfile path, open a temporary file and write the test data content with the scores in it
def print_score_tempfile(filepath, score):
    #print("Printing Score " + filepath)
    fp = tempfile.TemporaryFile()
    #no need to open a new file, rather use a temporary file
    #f_out = open(filepath + '.scored', 'w+')
    i=0
    for line in open(filepath):
        line = line.strip() + "\t" + str(score[i])
        i+=1
        print(line, file=fp)
        #print >> f_out, line
    return fp

#result should have three dimensions
#queryId, sentenceId, entity

def create_results_and_qrels(result):
    # [query_id(0), sentence_id(1), eq(2), sq(3), ec(4), sc(5), 0/1(6), sentence_similarity(7), score(8)]
    # print("converting data from " + filepath)
    f_qrel = open("results/qrel.txt", 'w+')
    f_results = open("results/results_multitask.txt", 'w+')
    for query_id in result.keys():
        # print(query_id)
        sentence_ids = result[query_id].keys()
        topk_sentences = []
        #total_positives = 0
        for sentence_id in sentence_ids:
            score = result[query_id][sentence_id]
            # score[0] is the score obtained for the sentence and score[1] is its original classification label
            print(query_id + "\t" + "0" +  "\t" + sentence_id + "\t" + str(score[1]), file=f_qrel)
            topk_sentences.append((sentence_id, score[0], score[1]))
            #total_positives += score[1]
        #retrieved_positives_at_topk = 0

        sorted_by_second = sorted(topk_sentences, key=lambda tup: tup[1], reverse=True)
        i = 1
        for sentence in topk_sentences:
            print(query_id + "\t" + "0" + "\t" + sentence[0] + "\t" + str(i) + "\t" + str(sentence[1]) + "\t" + "DNN", file=f_results)
            i+=1

        # print (final_result)


def extract_topk_sentences(result, topk):
    recall_at_topk = 0
    final_result = {}
    for query_id in result.keys():
        print(query_id)
        sentence_ids = result[query_id].keys()
        topk_sentences = []
        total_positives = 0
        for sentence_id in sentence_ids:
            score = result[query_id][sentence_id]
            #score[0] is the score obtained for the sentence and score[1] is its original classification label
            topk_sentences.append((sentence_id, score[0], score[1]))
            total_positives+=score[1]
        retrieved_positives_at_topk = 0
        sorted_by_second = sorted(topk_sentences, key=lambda tup: tup[1], reverse=True)
        i=0
        for tup in sorted_by_second:
            print(tup)
            if i<topk:
                retrieved_positives_at_topk+=tup[2]
            i+=1
        if total_positives==0:
            total_positives+=1
        recall_at_topk+=((1.0) * retrieved_positives_at_topk/total_positives)
        final_result[query_id] = sorted_by_second[0:topk]
    return final_result, (1.0 * recall_at_topk)/len(result.keys())

def compute_score_for_sentence(filepath):
    result = {}
    for line in open(filepath,'r'):
        l = line.strip().split("\t")
        #print (l)
        query_id = l[0].lower()
        sentence_id = l[1].lower()
        computed_score = float(l[8])
        #entity = l[4].lower()
        #if the query exists in the dictionary
        # if query_id in result.keys():
        #     #if the sentence exists in the dictionary
        #     if sentence_id in result[query_id].keys():
        #         #if the entity exists or not exists in the dictionary, it doesn't matter. overwrite or create new value
        #         if entity in result[query_id][sentence_id]:
        #             result[query_id][sentence_id][entity] = float(l[8])
        #         else:
        #             result[query_id][sentence_id][entity] = float(l[8])
        #
        #     else:
        #         result[query_id][sentence_id] = {}
        #         result[query_id][sentence_id][entity] = float(l[8])
        # else:
        #     result[query_id] = {}
        #     result[query_id] [sentence_id] = {}
        #     result[query_id][sentence_id] [entity] = float(l[8])

        #Alternative version with max
        if query_id in result.keys():
            #if the sentence exists in the dictionary
            if sentence_id in result[query_id].keys():
                #if the entity exists or not exists in the dictionary, it doesn't matter. overwrite or create new value
                # if entity in result[query_id][sentence_id]:
                #     result[query_id][sentence_id][entity] = float(l[8])
                # else:
                #     result[query_id][sentence_id][entity] = float(l[8])
                if float(l[8]) >= result[query_id][sentence_id][0]:
                    result[query_id][sentence_id] = (float(l[8]), int(l[6]))
            else:
                result[query_id][sentence_id] = (0, 0)
                #result[query_id][sentence_id][entity] = float(l[8])
        else:
            result[query_id] = {}
            result[query_id] [sentence_id] = (float(l[8]), int(l[6]))
            #result[query_id][sentence_id] [entity] = float(l[8])

    #return x2, x3, x4, x5, x6, x7
    return result

def compute_score_for_sentence_tempfile(filepath):
    result = {}
    for line in filepath:
        l = line.strip().split("\t")
        #print (l)
        query_id = l[0].lower()
        sentence_id = l[1].lower()
        computed_score = float(l[8])
        #entity = l[4].lower()
        #if the query exists in the dictionary
        # if query_id in result.keys():
        #     #if the sentence exists in the dictionary
        #     if sentence_id in result[query_id].keys():
        #         #if the entity exists or not exists in the dictionary, it doesn't matter. overwrite or create new value
        #         if entity in result[query_id][sentence_id]:
        #             result[query_id][sentence_id][entity] = float(l[8])
        #         else:
        #             result[query_id][sentence_id][entity] = float(l[8])
        #
        #     else:
        #         result[query_id][sentence_id] = {}
        #         result[query_id][sentence_id][entity] = float(l[8])
        # else:
        #     result[query_id] = {}
        #     result[query_id] [sentence_id] = {}
        #     result[query_id][sentence_id] [entity] = float(l[8])

        #Alternative version with max
        if query_id in result.keys():
            #if the sentence exists in the dictionary
            if sentence_id in result[query_id].keys():
                #if the entity exists or not exists in the dictionary, it doesn't matter. overwrite or create new value
                # if entity in result[query_id][sentence_id]:
                #     result[query_id][sentence_id][entity] = float(l[8])
                # else:
                #     result[query_id][sentence_id][entity] = float(l[8])
                if float(l[8]) >= result[query_id][sentence_id][0]:
                    result[query_id][sentence_id] = (float(l[8]), int(l[6]))
            else:
                result[query_id][sentence_id] = (0, 0)
                #result[query_id][sentence_id][entity] = float(l[8])
        else:
            result[query_id] = {}
            result[query_id] [sentence_id] = (float(l[8]), int(l[6]))
            #result[query_id][sentence_id] [entity] = float(l[8])

    #return x2, x3, x4, x5, x6, x7
    return result

def run_script(file_path, script_id):
    #filepath = "aquaint/validate/xaa.dat"
    #result = {}
    filepath = file_path
    #f_out = generate_random_score(filepath)
    #fp = tempfile.TemporaryFile()
    f_out = filepath + script_id + ".scored"
    result = compute_score_for_sentence(f_out)
    #print (len(result.keys()))
    final_result, recall = extract_topk_sentences(result, 100)
    #create_results_and_qrels(result)
    os.remove(f_out)
    return recall

def run_script_tempfile(file_path):
    #filepath = "aquaint/validate/xaa.dat"
    #result = {}
    #filepath = file_path
    #f_out = generate_random_score(filepath)
    #fp = tempfile.TemporaryFile()
    #f_out = filepath + ".scored"
    #file_path is a pointer to the temp file
    result = compute_score_for_sentence_tempfile(file_path)
    #print (len(result.keys()))
    final_result, recall = extract_topk_sentences(result, 20)
    create_results_and_qrels(result)
    #os.remove(f_out)
    return recall
