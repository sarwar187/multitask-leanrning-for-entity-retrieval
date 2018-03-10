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
    """
    creating a random score file for checking the performance of Neural Networks
    :param filepath:
    :return:
    """
    #print("Generating random score " + filepath)
    f_out = open(filepath + '.scored', 'w+')
    for line in open(filepath):
        line = line.strip() + "\t" + str(np.random.uniform())
        #print(line, file=f_out)
        #print >> f_out, line
    return filepath + '.scored'

def print_score(testfile, score, script_id):
    """
    take a single row from testfile, and concatenate the predicted score for it
    :param testfile: testfile. but for each script_id there will be a scored file
    :param score: score array that contains the predicted score
    :param script_id: script_id is for a specific parameter setting
    :return:
    """
    #print("Printing Score " + filepath)
    filepathfinal = testfile + script_id + ".scored"
    f_out = open(filepathfinal, 'w+')
    i=0
    sharefile = open(testfile,'r')
    for line in sharefile:
        line = line.strip() + "\t" + str(score[i])
        i+=1
        print(line, file=f_out)
        #print >> f_out, line
    f_out.close()
    sharefile.close()
    return filepathfinal


def create_results_and_qrels(result):
    """
    Creating results and qrels for applying traditional IR metrics
    :param result:
    :return:
    """

    # [query_id(0), sentence_id(1), eq(2), sq(3), ec(4), sc(5), 0/1(6), sentence_similarity(7), score(8)]
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
            #total_positives += score[1]
        #retrieved_positives_at_topk = 0

        topk_sentences.sort(key=lambda tup: tup[1], reverse=True)
        i = 1
        for sentence in topk_sentences:
            print(query_id + "\t" + "0" + "\t" + sentence[0] + "\t" + str(i) + "\t" + str(sentence[1]) + "\t" + "DNN", file=f_results)
            i+=1

        # print (final_result)


def extract_topk_sentences(result, topk, script_id=1):
    """
    This is for extracting the top-k sentences against each query_id
    It also computes recall in the top-k sentences
    :param result:
    :param topk:
    :return:
    """
    #pointers to qrel and results file
    f_qrel = open("results/qrel_" + str(script_id) + ".txt", 'w+')
    f_results = open("results/result_" + str(script_id) + ".txt", 'w+')

    recall_at_topk = 0
    final_result = {}
    for query_id in result.keys():
        print(query_id)
        sentence_ids = result[query_id].keys()
        sentence_score_label = []
        total_positives = 0
        for sentence_id in sentence_ids:
            predicted_score, original_label = result[query_id][sentence_id]
            print(query_id + "\t" + "0" + "\t" + sentence_id + "\t" + str(original_label), file=f_qrel)
            #score[0] is the score obtained for the sentence and score[1] is its original classification label
            sentence_score_label.append((sentence_id, predicted_score, original_label))
            total_positives+=original_label
        retrieved_positives_at_topk = 0
        #sorted_by_second = sorted(topk_sentences, key=lambda tup: tup[1], reverse=True)
        sentence_score_label.sort(key=lambda tup: tup[1], reverse=True)
        i=0
        # sentence_score_label contains (sentence_id, computed_score, original_label) tuples
        for tup in sentence_score_label:
            #print(tup)
            if i<topk:
                retrieved_positives_at_topk+=tup[2]
            print(query_id + "\t" + "0" + "\t" + tup[0] + "\t" + str(i) +
                  "\t" + str(tup[1]) + "\t" + "DNN", file=f_results)
            i+=1

        if total_positives==0:
            total_positives+=1
        recall_at_topk+=((1.0) * retrieved_positives_at_topk/total_positives)
        final_result[query_id] = sentence_score_label[0:topk]

    return final_result, (1.0 * recall_at_topk)/len(result.keys())


def compute_score_for_sentence(scorefile):
    """
    :param scorefile: scorefile contains (test_data, score) tuples
    :return: result array computed from all the test results using one hyperparameter setting
    """
    result = {}
    #entity_list = dict()
    for line in open(scorefile,'r'):
        l = line.strip().split("\t")
        query_id = l[0].lower()
        sentence_id = l[1].lower()
        computed_score = float(l[8])
        #computed_average_embedding_score = cosine_similarity(get_embedding())
        original_label = int(l[6])
        if query_id in result.keys():
            #if the sentence exists in the dictionary
            if sentence_id in result[query_id].keys():
                #storing a tuple against query_id, sentence_id pairs
                if float(computed_score) >= result[query_id][sentence_id][0]:
                    if original_label == 1:
                        result[query_id][sentence_id] = (computed_score, original_label)
                    else:
                        result[query_id][sentence_id] = (computed_score, result[query_id][sentence_id][1])
            else:
                result[query_id][sentence_id] = (computed_score, original_label)
        else:
            result[query_id] = {}
            result[query_id] [sentence_id] = (computed_score, original_label)
    return result


def run_script(testfile, script_id):
    """
    from a testfile location find the scored file for a specific hyperparameter setting
    then compute result dictionary (query_id, sentence_id, (original_loabel, predicted score))
    :param testfile:
    :param script_id:
    :return:
    """
    scorefile = testfile + script_id + ".scored"
    #create result dictionary from scorefile produced by a specific hyperparameter setting
    result = compute_score_for_sentence(scorefile)
    #from the results dictionary create quel and results file and compute recall in top-100 sentences
    final_result, recall = extract_topk_sentences(result, 1000, script_id)
    #remove the scorefile after doing all this things
    os.remove(scorefile)
    return recall, final_result

