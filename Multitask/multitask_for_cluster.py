from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import sys
from input_helpers import InputHelper
from evaluation_script import print_score
from evaluation_script import print_score_tempfile
from evaluation_script import run_script
from evaluation_script import run_script_tempfile
import argparse
import glob, os
import tempfile



inpH = InputHelper()

def load_data(data_path, to_dense=True):
    """Load data from the compressed NPZ file.

    Arguments:
        data_path: path to the data file.
        to_dense: if X is in the coordinate sparse format (COO), convert it
            to a dense ndarray when setting to_dense=True.

    Returns:
        X (numpy ndarray, shape = (samples, 3600)):
            Training input matrix where each row is a feature vector.
        y (numpy ndarray, shape = (samples,)):
            Training labels. Each entry is either 0 or 1.
        offset (numpy ndarray, shape = (samples, 2)):
            The (y, x) coordinate of the top-left corner of the
            28x28 bounding box where the MNIST digit is located.
    """
    data = np.load(data_path,encoding='latin1')
    X = data['X']
    if X.size == 1:
        X = data['X'][()]
        if to_dense:
            X = X.toarray()
    y = data['y']
    offset = data['offset']
    return X, y, offset

torch.manual_seed(0)
np.random.seed(0)

def load_word2_vec(emb_wiki, emb_fast):
    #inpH.loadW2VFloat(emb_path='/home/smsarwar/work/fastText/build/model_mincount5.vec', type="text")
    inpH.loadW2VFloat(emb_path=emb_wiki, type="text")
    #'merging begins'
    inpH.loadW2VFloat(emb_path=emb_fast, type="text")
    #inpH.loadW2VFloat(emb_path='/home/smsarwar/work/fastText/build/wiki.simple.vec', type="text")
    #print (len(inpH.pre_emb))

def average_embedding(tokens):
    #print(len(inpH.pre_emb.keys()))
    embeddings = np.asarray([inpH.pre_emb[token] for token in tokens if token in inpH.pre_emb])
    if len(embeddings) == 0:
        embeddings = np.zeros((1, 300))
    #print(embeddings)
    #print(embeddings.shape)
    return np.average(embeddings, axis=0)

def get_embedding(x2, x3, x4, x5):
    query_entity_embedding = average_embedding(x2.strip().split(" "))
    query_sentence_embedding = average_embedding(x3.strip().split(" "))
    candidate_entity_embedding = average_embedding(x4.strip().split(" "))
    candidate_sentence_embedding = average_embedding(x5.strip().split(" "))
    return np.concatenate((query_entity_embedding, query_sentence_embedding, candidate_entity_embedding, candidate_sentence_embedding), axis=0)

def convert_file(filepath):
    #print("converting data from " + filepath)
    X = []
    y_entity = []
    y_sentence = []
    for line in open(filepath):
        l = line.strip().split("\t")
        if len(l) < 5:
            print('error in data')
            continue
        x2 = l[2].lower()
        x3 = l[3].lower()
        x4 = l[4].lower()
        x5 = l[5].lower()
        x6 = l[6]
        x7 = l[7]
        embedding = get_embedding(x2, x3, x4, x5)
        #print ('embedding')
        X.append(embedding)
        y_entity.append(float(x6))
        y_sentence.append(float(x7))
    return np.asarray(X), np.asarray(y_entity), np.asarray(y_sentence)


class NN(nn.Module):
    def __init__(self, epochs=5, learning_rate=0.0001, weight_decay=0.9, alpha = 0.5, dropout_prb = 0.3, pcuda=0):
        super(NN, self).__init__()

        # Initializing model hyperparameters
        self.epochs = epochs
        self.learning_rate=learning_rate
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.dropout_prb = dropout_prb
        self.pcuda = pcuda
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
        X2 = Variable(torch.from_numpy(X).float())
        if self.pcuda == 1:
            X2 = X2.cuda()
        y_entity2 = Variable(torch.from_numpy(y_entity[:, np.newaxis]).float()).squeeze().view(y_entity.shape[0],1)
        if self.pcuda == 1:
           y_entity2 = y_entity2.cuda()

        y_sentence2 = Variable(torch.from_numpy(y_sentence[:, np.newaxis]).float()).squeeze()
        if self.pcuda == 1:
            y_sentence2 = y_sentence2.cuda()

        layer1 = F.dropout(F.relu(self.fclayer1(X2)), self.dropout_prb, training=self.training)      # 1st hidden layer
        if self.pcuda == 1:
            layer1 = layer1.cuda()

        layer2 = F.dropout(F.relu(self.fclayer2(layer1)), self.dropout_prb, training=self.training)  # 2nd hidden layer
        if self.pcuda == 1:
            layer2 = layer2.cuda()

        layer3 = F.dropout(F.relu(self.fclayer3(layer2)), self.dropout_prb, training=self.training)  # 3rd hidden layer
        if self.pcuda == 1:
            layer3 = layer3.cuda()

        y_entity_hat = self.fcEntity(layer3)      # class score
        if self.pcuda == 1:
            y_entity_hat = y_entity_hat.cuda()

        y_sentence_hat = F.softmax(self.fcSentence(layer3))  # class score

        if self.pcuda == 1:
            y_sentence_hat = y_sentence_hat.cuda()

        cross_entropy_loss = nn.BCEWithLogitsLoss(size_average=False)(input=y_entity_hat,
                                                                      target=y_entity2)  # cross entropy loss
        if self.pcuda == 1:
            cross_entropy_loss = cross_entropy_loss.cuda()

        mse_loss = nn.MSELoss(size_average=False)(input=y_sentence_hat, target=y_sentence2)  #mse loss for sentence, as it is a regression problem
        if self.pcuda == 1:
            mse_loss = mse_loss.cuda()

        self.loss = self.alpha * cross_entropy_loss + (1-self.alpha) * mse_loss
        if self.pcuda == 1:
            self.loss = self.loss.cuda()


        return self.loss

    def predict(self, X):
        X2 = Variable(torch.from_numpy(X).float())
        layer1 = F.relu(self.fclayer1(X2))  # 1st hidden layer
        layer2 = F.relu(self.fclayer2(layer1))  # 2nd hidden layer
        layer3 = F.relu(self.fclayer3(layer2))  # 3rd hidden layer
        y_entity_hat = self.fcEntity(layer3)  # class score
        y_sentence_hat = F.softmax(self.fcSentence(layer3))  # class score
        y_predict = F.sigmoid(y_entity_hat).data.numpy()
        y_predict = np.around(y_predict)
        #y_predict = np.argmax(y_predict, axis=1)
        return y_predict.ravel()

    def predict_score(self, X):
        X2 = Variable(torch.from_numpy(X).float())
        layer1 = F.relu(self.fclayer1(X2))  # 1st hidden layer
        layer2 = F.relu(self.fclayer2(layer1))  # 2nd hidden layer
        layer3 = F.relu(self.fclayer3(layer2))  # 3rd hidden layer
        y_entity_hat = self.fcEntity(layer3)  # class score
        y_predict = F.sigmoid(y_entity_hat).data.numpy()
        return y_predict.ravel()

    def fit(self, train_X, train_y_entity, train_y_sentence, test_X, test_y_entity, test_y_sentence):
        """Train the model according to the given training data.

        Arguments:
            X (numpy ndarray, shape = (samples, 3600)):
                Training input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Training labels. Each entry is either 0 or 1.
            y_loc (numpy ndarray, shape = (samples, 2)):
                Training (vertical, horizontal) locations of the
                objects.
        """

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        best_loss, best_test_loss = np.inf, np.inf
        best_epoch = 0
        batch_size = 250
        batches = range(int(train_X.shape[0] / batch_size))

        for epoch in np.arange(self.epochs):
            optimizer.zero_grad()
            for i in batches:
                nowX = train_X[i * batch_size:(i + 1)*batch_size]
                nowy_entity = train_y_entity[i * batch_size:(i + 1) * batch_size]
                nowy_sentence = train_y_sentence[i * batch_size:(i + 1) * batch_size]
                self.loss = self.forward(nowX, nowy_entity, nowy_sentence)
                self.loss.backward()  # backprop
                optimizer.step()
            self.loss = self.forward(train_X, train_y_entity, train_y_sentence)
            self.loss_test = self.forward(test_X, test_y_entity, test_y_sentence)

            if self.loss.data[0] < best_loss:
                best_loss = self.loss.data[0]
                best_epoch = epoch
                best_test_loss = self.loss_test.data[0]
                w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = self.get_model_params()


            #print('==> epoch {:.0f} -- train loss: {:.6f}, test loss: {:.6f}'.format(epoch, self.loss.data[0], self.loss_test.data[0]))
            #print('Train Accuracy: ', (self.predict(train_X) == train_y_entity).mean(), 'Test Accuracy: ', (self.predict(test_X) == test_y_entity).mean())

        #print('------------ ==> epoch {:.0f} -- best train loss: {:.6f}, best test loss: {:.6f}'.format(best_epoch, best_loss, best_test_loss))
        return w1, b1, w2, b2, w3, b3, w4, b4, w5, b5

    def get_model_params(self):
        """Get the parameters of the model.

        Returns:
            w1 (numpy ndarray, shape = (3600, 256)):
            b1 (numpy ndarray, shape = (256,)):
                weights and bias for FC(3600, 256)

            w2 (numpy ndarray, shape = (256, 64)):
            b2 (numpy ndarray, shape = (64,)):
                weights and bias for FC(256, 64)

            w3 (numpy ndarray, shape = (64, 32)):
            b3 (numpy ndarray, shape = (32,)):
                weights and bias for FC(64, 32)

            w4 (numpy ndarray, shape = (32, 10)):
            b4 (numpy ndarray, shape = (10,)):
                weights and bias for FC(32, 10) for class outputs
        """
        return self.fclayer1.weight.cpu().data.numpy().T, self.fclayer1.bias.cpu().data.numpy().ravel(), \
               self.fclayer2.weight.cpu().data.numpy().T, self.fclayer2.bias.cpu().data.numpy().ravel(), \
               self.fclayer3.weight.cpu().data.numpy().T, self.fclayer3.bias.cpu().data.numpy().ravel(), \
               self.fcEntity.weight.cpu().data.numpy().T, self.fcEntity.bias.cpu().data.numpy().ravel(), \
               self.fcSentence.weight.cpu().data.numpy().T, self.fcSentence.bias.cpu().data.numpy().ravel()

    def set_model_params(self, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
        """Set the parameters of the model.

        Arguments:
            w1 (numpy ndarray, shape = (3600, 256)):
            b1 (numpy ndarray, shape = (256,)):
                weights and bias for FC(3600, 256)

            w2 (numpy ndarray, shape = (256, 64)):
            b2 (numpy ndarray, shape = (64,)):
                weights and bias for FC(256, 64)

            w3 (numpy ndarray, shape = (64, 32)):
            b3 (numpy ndarray, shape = (32,)):
                weights and bias for FC(64, 32)

            w4 (numpy ndarray, shape = (32, 2)):
            b4 (numpy ndarray, shape = (2,)):
                weights and bias for FC(32, 2) for location outputs

            w5 (numpy ndarray, shape = (32, 1)):
            b5 (float):
                weights and bias for FC(32, 1) for the logit for
                class probability output
        """
        # print(w1.shape)
        # print(b1.shape)
        # print(w2.shape)
        # print(b2.shape)
        # print(w3.shape)
        # print(b3.shape)
        # print(w4.shape)
        # print(b4.shape)
        # print(w5.shape)
        # print(b5.shape)

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

def get_train_flie_list(directory):
    prefix = directory
    os.chdir(prefix)
    file_list = []
    for file in glob.glob("x*.dat"):
        file_list.append(prefix + file)
    #as we have moved two level inside, we need to go back two levels
    os.chdir("../../")
    #os.chdir(current_directory)
    return file_list


def main():
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
    #for train_file in file_list:
    #loop over the train files
    #train_X, train_entity, train_sentence = convert_file("torch_small/train.dat")
    train_X, train_entity, train_sentence = convert_file(train_file)
    #get file from the test folder
    #filepath = "torch_small/xaa.dat"
    validation_X, validation_entity, validation_sentence = convert_file(validation_file)
    #print(train_entity.shape)
    test_X, test_entity, test_sentence = convert_file(test_file)

    epochs = args.epochs
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    np.random.seed(0)
    for alpha in alphas:
        model = NN(epochs=epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay, alpha=alpha, pcuda=args.cuda)
        if args.cuda==1:
            model.cuda()
        w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = model.fit(train_X, train_entity, train_sentence, validation_X, validation_entity, validation_sentence)
        model = NN(epochs=epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay, alpha=alpha, pcuda=0)
        model.set_model_params(w1, b1, w2, b2, w3, b3, w4, b4, w5, b5)
        scores = model.predict_score(test_X)
        print_score(test_file, scores, args.script_id)
        recall_value = run_script(test_file, args.script_id)
        print(str(args.learning_rate), "\t" + str(args.weight_decay) + "\t" + str(alpha) + "\t" + str(recall_value))

if __name__ == '__main__':
    main()
