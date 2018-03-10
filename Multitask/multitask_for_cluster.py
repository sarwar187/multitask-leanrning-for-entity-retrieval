from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from input_helpers import InputHelper
from evaluation_script import print_score
from evaluation_script import run_script
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import argparse
import glob, os
import torchUtils
import socket
import re


torch.manual_seed(0)
np.random.seed(0)

inpH = InputHelper()

def load_word2_vec(emb_wiki, emb_fast):
    """ load word_2_vec from wiki and aquaint file paths. aquaint embedding have been created from
    fast text embedding
    :param emb_wiki: 300 dimensional embedding from wikipedia
    :param emb_fast: 300 dimensional embedding from aquaint
    :return: null. it completes the pre_emb array in InputHelper class
    """
    #load all the word_embedding files
    inpH.loadW2VFloat(emb_path=emb_fast, type="text")
    inpH.loadW2VFloat(emb_path=emb_wiki, type="text")
    #inpH.mergeW2V(emb_path=emb_wiki, type="text")

def average_embedding(tokens):
    """
    return the word embedding of a set of tokens. used for average sentence embedding
    :param tokens:
    :return: average of word embedding for a set of tokens
    """
    regex = re.compile(r"""
    (?<!\S)   # Assert there is no non-whitespace before the current character
    (?:       # Start of non-capturing group:
     [^\W\d_] # Match either a letter
     [\w-]*   # followed by any number of the allowed characters
    |         # or
     \d+      # match a string of digits.
    )         # End of group
    (?!\S)    # Assert there is no non-whitespace after the current character""",
    re.VERBOSE)

    embeddings = np.asarray([inpH.pre_emb[token] for token in tokens if token in inpH.pre_emb and regex.match(token)])
    #if we dont find embedding for any token in a sentence set all to zero
    if len(embeddings) == 0:
        embeddings = np.zeros((1, 300))
    return np.average(embeddings, axis=0)

def get_embedding(x2, x3, x4, x5):
    """
    needs to be re-written. too much tailored.
    :param x2: query entity embedding
    :param x3: query sentence embedding
    :param x4: candidate entity embedding
    :param x5: candidate sentence embedding
    :return: concatenated embedding vector of all the embeddings
    """
    query_entity_embedding = average_embedding(x2.strip().split(" "))
    query_sentence_embedding = average_embedding(x3.strip().split(" "))
    candidate_entity_embedding = average_embedding(x4.strip().split(" "))
    candidate_sentence_embedding = average_embedding(x5.strip().split(" "))
    return np.concatenate((query_entity_embedding, query_sentence_embedding, candidate_entity_embedding, candidate_sentence_embedding), axis=0)

def get_query_candidate_similarity(x2, x3, x4, x5):
    query = x2 + " " + x3
    candidate = x4 + " " + x5
    query_embedding = average_embedding(query.strip().split(" "))
    candidate_embedding = average_embedding(candidate.strip().split(" "))
    return cosine(query_embedding, candidate_embedding)

def convert_file(filepath):
    """
    trecqalist_215.7_1000.crfsuite  36487   guinevere       .       other film at the       new york festival , like francisco aliwalas ' `` disoriented '' and rea tajiri 's `` strawberry fields , '' also address issue relate to grow up in a multicultural society .   0       0.187124475837
    query_id, document_id, query entity, query sentence (w/o query entity), candidate entity
    :param filepath:
    :return:
    """
    X = []
    y_entity = []
    y_sentence = []
    baseline_scores = []
    for line in open(filepath):
        l = line.strip().split("\t")
        if len(l) < 5:
            print('error in data')
            continue
        x2 = l[2].lower()   #query entity embedding
        x3 = l[3].lower()   #query sentence embedding
        x4 = l[4].lower()   #candidate entity embedding
        x5 = l[5].lower()   #candidate sentence embedding
        x6 = l[6]           #entity score
        x7 = l[7]           #sentence similarity score
        embedding = get_embedding(x2, x3, x4, x5)
        # creating the data to be loaded in the neural model
        X.append(embedding)
        # loading entity ground truths
        y_entity.append(float(x6))
        # loading weak supervised entity similarity
        y_sentence.append(float(x7))
        # append cosine similarity score (baseline)
        baseline_scores.append(get_query_candidate_similarity(x2, x3, x4, x5))

    return np.asarray(X), np.asarray(y_entity), np.asarray(y_sentence), np.asarray(baseline_scores)

def convert_test_file(filepath):
    """
    trecqalist_215.7_1000.crfsuite  36487   guinevere       .       other film at the       new york festival , like francisco aliwalas ' `` disoriented '' and rea tajiri 's `` strawberry fields , '' also address issue relate to grow up in a multicultural society .   0       0.187124475837
    query_id, document_id, query entity, query sentence (w/o query entity), candidate entity
    :param filepath:
    :return:
    """
    #everything is lower-cased in the dataset
    test_entities = {}
    X = []
    y_entity = []
    y_sentence = []
    baseline_scores = []
    for line in open(filepath):
        l = line.strip().split("\t")

        if len(l) < 5:
            print('error in data')
            continue
        query_id = l[0]
        sentence_id = l[1]
        query_entity = l[2].lower()   #query entity string
        query_context = l[3].lower()   #query sentence string
        candidate_entity = l[4].lower()   #candidate entity string
        candidate_context = l[5].lower()   #candidate sentence string
        entity_similarity_label = l[6]           #entity score
        sentence_similarity_score = l[7]           #sentence similarity score
        embedding = get_embedding(query_entity, query_context, candidate_entity, candidate_context)
        # creating the data to be loaded in the neural model
        X.append(embedding)
        # loading entity ground truths
        y_entity.append(float(entity_similarity_label))
        # loading weak supervised entity similarity
        y_sentence.append(float(sentence_similarity_score))
        # append cosine similarity score (baseline)
        baseline_scores.append(get_query_candidate_similarity(query_entity, query_context, candidate_entity, candidate_context))
        if query_id in test_entities.keys():
            if sentence_id in test_entities[query_id].keys():
                test_entities[query_id][sentence_id].append(query_entity)
            else:
                test_entities[query_id][sentence_id] = []
                test_entities[query_id][sentence_id].append(query_entity)
        else:
            test_entities[query_id] = {}
            test_entities[query_id][sentence_id] = []
            test_entities[query_id][sentence_id].append(query_entity)

    return np.asarray(X), np.asarray(y_entity), np.asarray(y_sentence), np.asarray(baseline_scores),test_entities


class NN(nn.Module):
    def __init__(self, epochs=5, learning_rate=0.0001, weight_decay=0.9, alpha = 0.5, dropout_prb = 0.3, pcuda=0):
        super(NN, self).__init__()
        # Initializing model hyperparameters
        self.epochs = epochs
        self.learning_rate=learning_rate
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.dropout_prb = dropout_prb
        # Initializing 3 hidden layers
        self.fclayer1 = nn.Linear(1200, 256)
        #init.xavier_normal(self.fclayer1, gain=np.sqrt(2))
        self.fclayer2 = nn.Linear(256, 64)
        #init.xavier_normal(self.fclayer2, gain=np.sqrt(2))
        self.fclayer3 = nn.Linear(64, 32)
        #init.xavier_normal(self.fclayer3, gain=np.sqrt(2))
        self.fcSentence = nn.Linear(32, 1)
        self.fcEntity = nn.Linear(32, 1)
        #self.fcSentence = nn.Linear(32, 1)

    def forward(self, X, y_entity, y_sentence):
        """
        Network to learn entity similarity
        :param X:
        :param y_entity:
        :param y_sentence:
        :return:
        """
        #user torchUtils to convert directly to tensors
        X2= torchUtils.to_tensor(X)
        #print (X2)
        y_entity2 = torchUtils.to_tensor(y_entity[:, np.newaxis])
        y_sentence2 = torchUtils.to_tensor(y_sentence[:, np.newaxis])
        layer1 = F.dropout(F.relu(self.fclayer1(X2)), self.dropout_prb, training=self.training)      # 1st hidden layer
        layer2 = F.dropout(F.relu(self.fclayer2(layer1)), self.dropout_prb, training=self.training)  # 2nd hidden layer
        layer3 = F.dropout(F.relu(self.fclayer3(layer2)), self.dropout_prb, training=self.training)  # 3rd hidden layer
        y_entity_hat = self.fcEntity(layer3)      # class score
        y_sentence_hat = F.softmax(self.fcSentence(layer3))  # class score
        cross_entropy_loss = nn.BCEWithLogitsLoss(size_average=False)(input=y_entity_hat, target=y_entity2)  # cross entropy loss
        mse_loss = nn.MSELoss(size_average=False)(input=y_sentence_hat, target=y_sentence2)  #mse loss for sentence, as it is a regression problem
        #print(cross_entropy_loss)
        self.loss = self.alpha * cross_entropy_loss + (1-self.alpha) * mse_loss
        return self.loss

    def predict(self, X):
        """
        :param X:
        :return:
        """
        X2 = Variable(torch.from_numpy(X).float())
        layer1 = F.relu(self.fclayer1(X2))      # 1st hidden layer
        layer2 = F.relu(self.fclayer2(layer1))  # 2nd hidden layer
        layer3 = F.relu(self.fclayer3(layer2))  # 3rd hidden layer
        y_entity_hat = self.fcEntity(layer3)    # class score
        y_sentence_hat = F.softmax(self.fcSentence(layer3))  # class score
        y_predict = F.sigmoid(y_entity_hat).data.numpy()
        y_predict = np.around(y_predict)
        #y_predict = np.argmax(y_predict, axis=1)
        return y_predict.ravel()

    def predict_score(self, X):
        """

        :param X:
        :return:
        """
        X2 = Variable(torch.from_numpy(X).float())
        layer1 = F.relu(self.fclayer1(X2))  # 1st hidden layer
        layer2 = F.relu(self.fclayer2(layer1))  # 2nd hidden layer
        layer3 = F.relu(self.fclayer3(layer2))  # 3rd hidden layer
        y_entity_hat = self.fcEntity(layer3)  # class score
        y_predict = F.sigmoid(y_entity_hat).data.numpy()
        return y_predict.ravel()

    def fit(self, train_X, train_y_entity, train_y_sentence, validation_X, validation_y_entity, validation_y_sentence):
        """Train the model according to the given training data.

        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        best_loss, best_test_loss = np.inf, np.inf
        best_epoch = 0
        batch_size = 250
        batches = range(int(train_X.shape[0] / batch_size))

        for epoch in np.arange(self.epochs):
            optimizer.zero_grad()
            for i in batches:
                nowX = train_X[i * batch_size:(i + 1) * batch_size]
                nowy_entity = train_y_entity[i * batch_size:(i + 1) * batch_size]
                nowy_sentence = train_y_sentence[i * batch_size:(i + 1) * batch_size]
                loss_train = self.forward(nowX, nowy_entity, nowy_sentence).data[0]
                self.loss.backward()  # backprop
                optimizer.step()
                #self.loss = self.forward(train_X, train_y_entity, train_y_sentence)
                loss_validation = self.forward(validation_X, validation_y_entity, validation_y_sentence).data[0]

            if loss_validation < best_loss:
                best_loss = loss_validation
                best_epoch = epoch
                best_train_loss = loss_train
                w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = self.get_model_params()

            print('==> epoch {:.0f} -- train loss: {:.3f}, validation loss: {:.3f}'.format(epoch, loss_train, loss_validation))
            #print('Train Accuracy: ', (self.predict(train_X) == train_y_entity).mean(), 'Test Accuracy: ', (self.predict(test_X) == test_y_entity).mean())

        #print('------------ ==> epoch {:.0f} -- best train loss: {:.6f}, best test loss: {:.6f}'.format(best_epoch, best_loss, best_test_loss))
        return w1, b1, w2, b2, w3, b3, w4, b4, w5, b5

    def get_model_params(self):
        """Get the parameters of the model.
        """
        return self.fclayer1.weight.cpu().data.numpy().T, self.fclayer1.bias.cpu().data.numpy().ravel(), \
               self.fclayer2.weight.cpu().data.numpy().T, self.fclayer2.bias.cpu().data.numpy().ravel(), \
               self.fclayer3.weight.cpu().data.numpy().T, self.fclayer3.bias.cpu().data.numpy().ravel(), \
               self.fcEntity.weight.cpu().data.numpy().T, self.fcEntity.bias.cpu().data.numpy().ravel(), \
               self.fcSentence.weight.cpu().data.numpy().T, self.fcSentence.bias.cpu().data.numpy().ravel()

    def set_model_params(self, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
        """Set the parameters of the model.
        """
        self.fclayer1.weight = torch.nn.Parameter((torch.from_numpy(w1.T)))
        self.fclayer2.weight = torch.nn.Parameter((torch.from_numpy(w2.T)))
        self.fclayer3.weight = torch.nn.Parameter((torch.from_numpy(w3.T)))
        self.fcEntity.weight = torch.nn.Parameter((torch.from_numpy(w4.T)))
        self.fcSentence.weight = torch.nn.Parameter((torch.from_numpy(w5.T)))
        self.fclayer1.bias = torch.nn.Parameter((torch.from_numpy(b1)))
        self.fclayer2.bias = torch.nn.Parameter((torch.from_numpy(b2)))
        self.fclayer3.bias = torch.nn.Parameter((torch.from_numpy(b3)))
        self.fcEntity.bias = torch.nn.Parameter((torch.from_numpy(b4)))
        self.fcSentence.bias = torch.nn.Parameter((torch.from_numpy(b5)))


def main():
    #print(socket.gethostname())
    parser = argparse.ArgumentParser(description="parameters include cuda")
    parser.add_argument('epochs', type=int, help='please use number of epochs')
    parser.add_argument('learning_rate', type=float, help='please use a learning rate')
    parser.add_argument('weight_decay', type=float, help='please use regularization')
    parser.add_argument('cuda', type=int, help='if you want to use GPU pass this as 1')
    parser.add_argument('dummy', type=int, help='if you want to use small dataset pass 1')
    parser.add_argument('emb_wiki', type=str, help='if you want to use small dataset pass 1')
    parser.add_argument('emb_fast', type=str, help='if you want to use small dataset pass 1')
    parser.add_argument('script_id', type=str, help='if you want to use small dataset pass 1')

    args = parser.parse_args()
    load_word2_vec(args.emb_wiki, args.emb_fast)

    if socket.gethostname() == "brooloo":
        directory = "/home/smsarwar/PycharmProjects/deep-siamese-text-similarity/aquaint/"
        dummy_directory = "/home/smsarwar/PycharmProjects/deep-siamese-text-similarity/aquaint/dummy/"
    else:
        directory = "aquaint/"
        dummy_directory = "aquaint/dummy/"

    if args.dummy == 1:
        train_file = dummy_directory + "train_sentence_sim.dat"
        test_file = dummy_directory + "test_sentence_sim.dat"
        validation_file = dummy_directory + "validation_sentence_sim.dat"
    else:
        train_file = directory + "train_sentence_sim.dat"
        test_file =  directory + "test_sentence_sim.dat"
        validation_file = directory + "validation_sentence_sim.dat"

    #after training and validation pick the best model and use that on the test data.
    train_X, train_entity, train_sentence, baseline_scores_train = convert_file(train_file)
    #get file from the test folder
    validation_X, validation_entity, validation_sentence, baseline_scores_validation = convert_file(validation_file)
    #print(train_entity.shape)
    test_X, test_entity, test_sentence, baseline_scores_test, test_entities = convert_test_file(test_file)


    print ('train_X shape {:s}, validation_X shape {:s}, test_X shape {:s}'.format(train_X.shape, validation_X.shape, test_X.shape))
    epochs = args.epochs
    #alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    alphas = [0.5]
    np.random.seed(0)
    for alpha in alphas:
        model = NN(epochs=epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay, alpha=alpha, pcuda=args.cuda)
        if args.cuda==1:
            model.cuda()
        #w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = model.fit(train_X, train_entity, train_sentence, validation_X, validation_entity, validation_sentence)

        #w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = model.fit(train_X, train_entity, train_sentence, test_X,
        #                                                   test_entity, test_sentence)

        #w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = model.fit(test_X, test_entity, test_sentence, train_X,
        #                                                  train_entity, train_sentence)

        #model = NN(epochs=epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay, alpha=alpha, pcuda=0)

        #model.set_model_params(w1, b1, w2, b2, w3, b3, w4, b4, w5, b5)

        #scores = model.predict_score(test_X)

        #scores = model.predict_score(train_X)

        #print score prints the results in a scored file identified by the script_id
        #for each script there will be scored file
        #print_score(test_file, scores, args.script_id)
        #print_score(train_file, scores, args.script_id)
        #from the test file create qrel, results and find the value of recall at topk
        #recall_value = run_script(test_file, args.script_id)

        #checking the performance of baseline
        print_score(test_file, baseline_scores_test, args.script_id)
        recall_value, final_result = run_script(test_file, args.script_id)

        for query_id in final_result.keys():
            seen_entity_set = set()
            query = query_id.split(".")
            result_file = open(dummy_directory + "evaluate/" + query[0] + "."+ query[1] + ".test.result", "w+")
            for sentence_score_label in final_result[query_id]:
                 #print (sentence_score_label[0])
                 #print (test_entities[query_id][sentence_score_label[0]])
                 #seen_entity_set.add(test_entities[query_id][sentence_score_label[0]])
                 print(sentence_score_label[0], file = result_file)
            result_file.close()
        #recall_value = run_script(train_file, args.script_id)
        print(str(args.learning_rate), "\t" + str(args.weight_decay) + "\t" + str(alpha) + "\t" + str(recall_value))

if __name__ == '__main__':
    main()
