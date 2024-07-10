import itertools
import random
import pandas as pd

class MyScheduler():
    def __init__(self, opt):
        self.activations = ['relu', 'sigmoid', 'hardtanh', 'softmax']
        self.optimizer_choices = ['Adam', 'SGD', 'Adagrad', 'RMSprop']   
        self.sum_epoch=0
        self.min_epoch=opt.epoch_min
        self.limit_epoch=opt.epoch_limit
        self.result = []
        self.batchSize=[16,32,64,128]
        self.epoch=[10,15,20]
        self.lr=[0.001,0.002,0.004,0.01,0.02]
        self.activation=[0,1,2,3]
        self.optimizer=[0,1,2,3]
        self.config = list(itertools.product(self.batchSize, self.epoch, self.lr, self.activation, self.optimizer))
        self.config.remove((opt.batchSize,opt.epoch,opt.lr,opt.activation,opt.optimizer))
        random.shuffle(self.config)

    def count_epoch(self, epoch):
        if self.sum_epoch + epoch >= self.limit_epoch:
            return True
        return False

    def eval(self, epoch, history):
        if self.count_epoch(epoch):
            return True
        if epoch <= self.min_epoch:
            return False
        '''
        if history["train_loss"][-1] > history["train_loss"][-2]:
            return True
        if history["validation_loss"][-1] > history["validation_loss"][-2]:
            return True
        '''
        if history["validation_acc"][-1] < history["validation_acc"][-2]:
            return True
        else:
            return False

    def search(self, index, epoch, opt, history):
        self.sum_epoch += epoch
        self.result.append([opt.batchSize, opt.epoch, opt.lr, self.activations[opt.activation], self.optimizer_choices[opt.optimizer], epoch, history["validation_acc"][-1]])
        if self.count_epoch(0):
            self.result = sorted(self.result, reverse=True, key=lambda x: x[6])
            df = pd.DataFrame(self.result, columns=['batchSize', 'epoch', 'learning rate', 'activation', 'optimizer', 'actual epoch', 'latest accuracy'])
            df.to_csv('./logs/optimize.txt', sep='\t')
            df.to_csv('./logs/optimize.csv', sep=',')
            opt.batchSize = self.result[0][0]
            opt.epoch = self.result[0][1]
            opt.lr = self.result[0][2]
            opt.activation = self.activations.index(self.result[0][3])
            opt.optimizer = self.optimizer_choices.index(self.result[0][4]) 
            return True
        opt.batchSize = self.config[index][0]
        opt.epoch = self.config[index][1]
        opt.lr = self.config[index][2]
        opt.activation = self.config[index][3]
        opt.optimizer = self.config[index][4]
        return False