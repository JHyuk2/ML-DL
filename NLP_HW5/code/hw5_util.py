# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import random
import math
import torch

from PyKomoran import *
from tqdm import trange
from sklearn.metrics import *

#   *** Do not modify the code ***
class NLP_util:
    def __init__(self):
        self.labels = sorted(['opening', 'request', 'wh-question', 'yn-question', 'inform', 'affirm', 'ack', 'expressive'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.komoran = Komoran('EXP')
        self.word2idx = dict()
        self.labels2idx = dict()
        self.idx2labels = dict()
        self.data_dict = None
        self.test_dataset = None
        self.train_dataset = None

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def convert_examples_to_features(self, data):
        return

    def create_w2i_l2i_i2l(self, data):
        return
        
    def make_dataset(self, input_ids, labels):
        return torch.utils.data.TensorDataset(torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long))

    def load_data(self, input_path=None):
        assert type(input_path) is str, '입력 파일 경로(input_path)를 확인하세요.'

        with open(input_path, mode='r', encoding='utf-8') as f:
            data = json.load(f)

            if input_path.split('_')[-1][:-5] == 'te':
                self.data_dict = data
                input_ids, labels = self.convert_examples_to_features(data)
                self.test_dataset = self.make_dataset(input_ids, labels)
            elif input_path.split('_')[-1][:-5] == 'tr':
                self.create_w2i_l2i_i2l(data)
                input_ids, labels = self.convert_examples_to_features(data)
                self.train_dataset = self.make_dataset(input_ids, labels)
            else:
                raise ValueError('입력파일이 train 파일인지, test 파일인지 확인이 불가능합니다.')

    def save_result(self, result, std_name, std_id):
        with open('./'+str(std_name)+'_'+str(std_id)+'_hw5.txt', mode='w', encoding='utf-8') as f:
            f.write('********** Eval Result **********\n')
            f.write('Macro averaging precision: {:.2f}%\n'.format(result['macro precision']))
            f.write('Micro averaging precision: {:.2f}%\n\n'.format(result['micro precision']))

            f.write('Macro averaging recall: {:.2f}%\n'.format(result['macro recall']))
            f.write('Micro averaging recall: {:.2f}%\n\n'.format(result['micro recall']))

            f.write('Macro averaging f1-score: {:.2f}%\n'.format(result['macro f1']))
            f.write('Micro averaging f1-score: {:.2f}%\n\n'.format(result['micro f1']))

            f.write('Dialogue Number: {}'.format(result['dialogue number']))
            
            for Utterance in result['matching']:
                f.write('\n\nUtterance: {}'.format(Utterance[0]))
                f.write('\nReal label: {}'.format(Utterance[1]))
                f.write('\nPredicted label: {}'.format(Utterance[2]))
                f.write('\nResult: {}'.format(Utterance[3]))
        f.close()
    
    def check_optimizer(self, model, optimizer_name, learning_rate):
        if optimizer_name == 'adam':
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError('optimizer_name이 pytorch에 존재하지 않습니다. 다시 확인하세요.')
    
    def check_loss_ft(self, loss_ft):
        if loss_ft == 'mseloss':
            return torch.nn.MSELoss()
        elif loss_ft == 'crossentropyloss':
            return torch.nn.CrossEntropyLoss()
        elif loss_ft == 'nllloss':
            return torch.nn.NLLLoss()
        else:
            raise ValueError('loss_function이 pytorch에 존재하지 않습니다. 다시 확인하세요.')

    def train(self, model, loss_ft, optimizer_name, learning_rate, train_batch_size, num_train_epochs):
        # optimizer
        assert type(optimizer_name) is str, 'optimizer_name의 type은 string이 되어야 합니다.'
        optimizer = self.check_optimizer(model, optimizer_name.lower(), learning_rate)

        # Loss function
        assert type(loss_ft) is str, 'loss_ft type은 string이 되어야 합니다.'
        criterion = self.check_loss_ft(loss_ft.lower())

        self.set_seed(42)
        
        train_DataLoader = torch.utils.data.DataLoader(self.train_dataset, shuffle=True, batch_size=train_batch_size)

        train_iterator = trange(num_train_epochs, desc="Epoch")

        print("\n***** Running training *****")
        print("  Num examples = {}".format(len(self.train_dataset)))
        print("  Num Epochs = {}".format(num_train_epochs))
        print("  Train Batch size = {}".format(train_batch_size))
        print("  Device = ", self.device)
        
        print('hello')
        
        model.to(self.device)
        model.train(True)
        model.zero_grad()
        for epoch in train_iterator:
            loss = 0
            for batch in train_DataLoader:
                input_vector = batch[0].to(self.device)
                label = batch[1].to(self.device)
                predict = model(input_vector)

                loss = criterion(predict, label)
                loss += loss.item()

                loss.backward()
                optimizer.step()
                model.zero_grad()

            if (epoch+1) % 50 == 0:
                print("\n********** Train Result **********")
                print("  Epoch / Total Epoch : {} / {}".format(epoch + 1, num_train_epochs))
                print("  Loss : {:.4f}".format(loss))
                
        model.train(False)

    def predict(self, model):
        test_DataLoader = torch.utils.data.DataLoader(self.test_dataset, shuffle=False, batch_size=1)

        print("***** Running Prediction *****")
        print("  Num examples = {}".format(len(self.test_dataset)))
        print("  Test Batch size = 1")

        model.eval()
        pred = None
        label = None
        for batch in test_DataLoader:
            input_vector = batch[0].to(self.device)

            with torch.no_grad():
                predict = model(input_vector)
            
            if pred is None:
                pred = predict.detach().cpu().numpy()
                label = batch[1].numpy()
            else:
                pred = np.append(pred, predict.detach().cpu().numpy(), axis=0)
                label = np.append(label, batch[1].numpy(), axis=0)

        pred = np.argmax(pred, axis=1)

        print("***** Prediction 완료 *****")

        return pred.tolist(), label.tolist()
    
    def eval(self, pred, label, dialogue_number):
        return
#   *** Do not modify the code ***