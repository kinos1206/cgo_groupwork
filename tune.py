import itertools
import random
import pandas as pd
from deap import base, creator, tools, algorithms

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

class MySchedulerGA():
    def __init__(self, opt):
        self.activations = ['relu', 'sigmoid', 'hardtanh', 'softmax']
        self.optimizer_choices = ['Adam', 'SGD', 'Adagrad', 'RMSprop']
        self.sum_epoch = 0
        self.min_epoch = opt.epoch_min
        self.limit_epoch = opt.epoch_limit
        self.result = []
        self.current_individual = None  # 現在の個体を保持する
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # 遺伝的アルゴリズムの設定
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_batch_size", random.choice, [16, 32, 64, 128])
        self.toolbox.register("attr_epoch", random.choice, [10, 15, 20])
        self.toolbox.register("attr_lr", random.uniform, 0.001, 0.02)
        self.toolbox.register("attr_activation", random.choice, [0, 1, 2, 3])
        self.toolbox.register("attr_optimizer", random.choice, [0, 1, 2, 3])

        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                              (self.toolbox.attr_batch_size,
                               self.toolbox.attr_epoch,
                               self.toolbox.attr_lr,
                               self.toolbox.attr_activation,
                               self.toolbox.attr_optimizer), n=1)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate)

    def count_epoch(self, epoch):
        return self.sum_epoch + epoch >= self.limit_epoch

    def eval(self, epoch, history):
        if self.count_epoch(epoch):
            return True
        if epoch <= self.min_epoch:
            return False
        if history["validation_acc"][-1] < history["validation_acc"][-2]:
            return True
        else:
            return False

    def update_result(self, acc):
        if self.current_individual is not None:
            batch_size, epoch, lr, activation, optimizer = self.current_individual
            self.result.append([batch_size, epoch, lr, self.activations[activation], self.optimizer_choices[optimizer], epoch, acc])

    def get_next(self, opt):
        if self.current_individual is None:
            self.current_individual = self.toolbox.individual()
        opt.batchSize, opt.epoch, opt.lr, opt.activation, opt.optimizer = self.current_individual
        return self.current_individual

    def evaluate(self, individual):
        batch_size, epoch, lr, activation, optimizer = individual
        return (random.uniform(0.8, 1.0),)

    def search(self, opt):
        # 現在の個体を評価
        acc = self.result[-1][-1] if self.result else 0
        self.update_result(acc)

        # 初期化
        population = self.toolbox.population(n=50)
        hof = tools.HallOfFame(1)

        # 遺伝的アルゴリズムの適用
        algorithms.eaSimple(population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=10, halloffame=hof, verbose=True)

        # 結果の保存と最適ハイパーパラメータの設定
        self.result = sorted(self.result, reverse=True, key=lambda x: x[6])
        df = pd.DataFrame(self.result, columns=['batchSize', 'epoch', 'learning rate', 'activation', 'optimizer', 'actual epoch', 'latest accuracy'])
        df.to_csv('./logs/optimize_ga.txt', sep='\t')
        df.to_csv('./logs/optimize_ga.csv', sep=',')

        best_ind = hof[0]
        opt.batchSize, opt.epoch, opt.lr, opt.activation, opt.optimizer = best_ind

        # 新しい個体を取得
        self.current_individual = self.toolbox.individual()
        return True if self.count_epoch(0) else False
