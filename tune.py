import itertools
import random

import pandas as pd

from .options import TOptions


# ランダムサーチ
class MyScheduler:
    def __init__(self, opt: TOptions) -> None:
        random.seed(0)  # for reproducibility

        self.activations = ['relu', 'sigmoid', 'hardtanh', 'softmax']
        self.optimizer_choices = ['Adam', 'SGD', 'Adagrad', 'RMSprop']
        self.sum_epoch = 0
        self.min_epoch = opt.epoch_min
        self.limit_epoch = opt.epoch_limit
        self.result = []
        self.batchSize = [16, 32, 64, 128]
        self.epoch = [10, 15, 20]
        self.lr = [0.001, 0.002, 0.004, 0.01, 0.02]
        self.activation = [0, 1, 2, 3]
        self.optimizer = [0, 1, 2, 3]
        self.config = list(itertools.product(self.batchSize, self.epoch, self.lr, self.activation, self.optimizer))
        self.config.remove((opt.batchSize, opt.epoch, opt.lr, opt.activation, opt.optimizer))
        random.shuffle(self.config)

    def count_epoch(self, epoch: int) -> bool:
        if self.sum_epoch + epoch >= self.limit_epoch:
            return True
        return False

    def eval(self, epoch: int, history: dict) -> bool:
        if self.count_epoch(epoch):
            return True
        if epoch <= self.min_epoch:
            return False
        """
        if history["train_loss"][-1] > history["train_loss"][-2]:
            return True
        if history["validation_loss"][-1] > history["validation_loss"][-2]:
            return True
        """
        if history['validation_acc'][-1] < history['validation_acc'][-2]:
            return True
        else:
            return False

    def search(self, index: int, epoch: int, opt: TOptions, history: dict) -> bool:
        self.sum_epoch += epoch
        self.result.append(
            [
                opt.batchSize,
                opt.epoch,
                opt.lr,
                self.activations[opt.activation],
                self.optimizer_choices[opt.optimizer],
                epoch,
                history['validation_acc'][-1],
            ]
        )

        if self.count_epoch(0):
            self.result = sorted(self.result, reverse=True, key=lambda x: x[6])
            df = pd.DataFrame(
                self.result,
                columns=[
                    'batchSize',
                    'epoch',
                    'learning rate',
                    'activation',
                    'optimizer',
                    'actual epoch',
                    'latest accuracy',
                ],
            )
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


class MySchedulerGA:
    def __init__(self, opt):
        self.activations = ['relu', 'sigmoid', 'hardtanh', 'softmax']
        self.optimizer_choices = ['Adam', 'SGD', 'Adagrad', 'RMSprop']
        self.population_size = 10
        self.mutation_rate = 0.1
        self.sum_epoch = 0
        self.min_epoch = opt.epoch_min
        self.limit_epoch = opt.epoch_limit
        self.result = []
        self.batchSize = [16, 32, 64, 128]
        self.epoch = [10, 15, 20]
        self.lr = [0.001, 0.002, 0.004, 0.01, 0.02]
        self.activation = [0, 1, 2, 3]
        self.optimizer = [0, 1, 2, 3]

        self.population = self.initialize_population(opt)

    def initialize_population(self, opt):
        population = []
        for _ in range(self.population_size):
            individual = {
                'batchSize': random.choice(self.batchSize),
                'epoch': random.choice(self.epoch),
                'lr': random.choice(self.lr),
                'activation': random.choice(self.activation),
                'optimizer': random.choice(self.optimizer),
            }
            population.append(individual)
        # Ensure the initial configuration is included
        population.append(
            {
                'batchSize': opt.batchSize,
                'epoch': opt.epoch,
                'lr': opt.lr,
                'activation': opt.activation,
                'optimizer': opt.optimizer,
            }
        )
        return population

    def evaluate_fitness(self, history):
        return history['validation_acc'][-1]

    def select_parents(self):
        parents = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        return parents[:2]  # Select top 2 individuals

    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1:
            child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            individual['batchSize'] = random.choice(self.batchSize)
        if random.random() < self.mutation_rate:
            individual['epoch'] = random.choice(self.epoch)
        if random.random() < self.mutation_rate:
            individual['lr'] = random.choice(self.lr)
        if random.random() < self.mutation_rate:
            individual['activation'] = random.choice(self.activation)
        if random.random() < self.mutation_rate:
            individual['optimizer'] = random.choice(self.optimizer)

    def count_epoch(self, epoch):
        if self.sum_epoch + epoch >= self.limit_epoch:
            return True
        return False

    def eval(self, epoch, history):
        if self.count_epoch(epoch):
            return True
        if epoch <= self.min_epoch:
            return False
        if history['validation_acc'][-1] < history['validation_acc'][-2]:
            return True
        else:
            return False

    def search(self, index, epoch, opt, history):
        self.sum_epoch += epoch
        fitness = self.evaluate_fitness(history)
        self.result.append(
            [
                opt.batchSize,
                opt.epoch,
                opt.lr,
                self.activations[opt.activation],
                self.optimizer_choices[opt.optimizer],
                epoch,
                fitness,
            ]
        )

        if self.count_epoch(0):
            self.result = sorted(self.result, reverse=True, key=lambda x: x[6])
            df = pd.DataFrame(
                self.result,
                columns=[
                    'batchSize',
                    'epoch',
                    'learning rate',
                    'activation',
                    'optimizer',
                    'actual epoch',
                    'latest accuracy',
                ],
            )
            df.to_csv('./logs/optimize.txt', sep='\t')
            df.to_csv('./logs/optimize.csv', sep=',')
            best_individual = self.result[0]
            opt.batchSize = best_individual[0]
            opt.epoch = best_individual[1]
            opt.lr = best_individual[2]
            opt.activation = self.activations.index(best_individual[3])
            opt.optimizer = self.optimizer_choices.index(best_individual[4])
            return True

        # 現在の集団の適応度を評価
        for individual in self.population:
            individual['fitness'] = fitness

        # 親の選択
        parent1, parent2 = self.select_parents()

        # 交叉と突然変異による新しい個体群の生成
        new_population = []
        for _ in range(self.population_size):
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)

        self.population = new_population

        # 新しいハイパーパラメータをoptに設定
        next_individual = self.population[0]
        opt.batchSize = next_individual['batchSize']
        opt.epoch = next_individual['epoch']
        opt.lr = next_individual['lr']
        opt.activation = next_individual['activation']
        opt.optimizer = next_individual['optimizer']

        return False


class MySchedulerPSO:
    def __init__(self, opt):
        self.activations = ['relu', 'sigmoid', 'hardtanh', 'softmax']
        self.optimizer_choices = ['Adam', 'SGD', 'Adagrad', 'RMSprop']
        self.population_size = 10
        self.sum_epoch = 0
        self.min_epoch = opt.epoch_min
        self.limit_epoch = opt.epoch_limit
        self.result = []
        self.c1 = 1.5  # 個体速度の学習係数
        self.c2 = 1.5  # 全体速度の学習係数
        self.w = 0.5  # 慣性係数
        self.batchSize = [16, 32, 64, 128]
        self.epoch = [10, 15, 20]
        self.lr = [0.001, 0.002, 0.004, 0.01, 0.02]
        self.activation = [0, 1, 2, 3]
        self.optimizer = [0, 1, 2, 3]

        self.population = self.initialize_population(opt)
        self.velocities = self.initialize_velocities()
        self.pbest = self.population.copy()
        self.gbest = None
        self.gbest_fitness = None

    def initialize_population(self, opt):
        population = []
        for _ in range(self.population_size):
            individual = {
                'batchSize': random.choice(self.batchSize),
                'epoch': random.choice(self.epoch),
                'lr': random.choice(self.lr),
                'activation': random.choice(self.activation),
                'optimizer': random.choice(self.optimizer),
            }
            population.append(individual)
        population.append(
            {
                'batchSize': opt.batchSize,
                'epoch': opt.epoch,
                'lr': opt.lr,
                'activation': opt.activation,
                'optimizer': opt.optimizer,
            }
        )
        return population

    def initialize_velocities(self):
        velocities = []
        for _ in range(self.population_size):
            velocity = {
                'batchSize': random.uniform(-1, 1),
                'epoch': random.uniform(-1, 1),
                'lr': random.uniform(-1, 1),
                'activation': random.uniform(-1, 1),
                'optimizer': random.uniform(-1, 1),
            }
            velocities.append(velocity)
        return velocities

    def evaluate_fitness(self, history):
        return history['validation_acc'][-1]

    def update_velocities_and_positions(self):
        for i in range(self.population_size):
            for key in self.population[i]:
                r1, r2 = random.random(), random.random()
                cognitive_velocity = self.c1 * r1 * (self.pbest[i][key] - self.population[i][key])
                if self.gbest is not None:
                    social_velocity = self.c2 * r2 * (self.gbest[key] - self.population[i][key])
                self.velocities[i][key] = self.w * self.velocities[i][key] + cognitive_velocity + social_velocity
                self.population[i][key] += self.velocities[i][key]

                # 値の制約 (必要に応じて調整)
                if key in ['batchSize', 'epoch']:
                    self.population[i][key] = int(
                        round(max(min(self.population[i][key], max(getattr(self, key))), min(getattr(self, key))))
                    )
                elif key == 'lr':
                    self.population[i][key] = max(min(self.population[i][key], max(self.lr)), min(self.lr))
                elif key in ['activation', 'optimizer']:
                    self.population[i][key] = int(
                        round(max(min(self.population[i][key], max(getattr(self, key))), min(getattr(self, key))))
                    )

    def count_epoch(self, epoch):
        if self.sum_epoch + epoch >= self.limit_epoch:
            return True
        return False

    def eval(self, epoch, history):
        if self.count_epoch(epoch):
            return True
        if epoch <= self.min_epoch:
            return False
        if history['validation_acc'][-1] < history['validation_acc'][-2]:
            return True
        else:
            return False

    def search(self, index, epoch, opt, history):
        self.sum_epoch += epoch
        fitness = self.evaluate_fitness(history)
        self.result.append(
            [
                opt.batchSize,
                opt.epoch,
                opt.lr,
                self.activations[opt.activation],
                self.optimizer_choices[opt.optimizer],
                epoch,
                fitness,
            ]
        )

        if self.gbest is None or fitness > self.gbest_fitness:
            self.gbest = self.population[index].copy()
            self.gbest_fitness = fitness

        if self.count_epoch(0):
            self.result = sorted(self.result, reverse=True, key=lambda x: x[6])
            df = pd.DataFrame(
                self.result,
                columns=[
                    'batchSize',
                    'epoch',
                    'learning rate',
                    'activation',
                    'optimizer',
                    'actual epoch',
                    'latest accuracy',
                ],
            )
            df.to_csv('./logs/optimize.txt', sep='\t')
            df.to_csv('./logs/optimize.csv', sep=',')
            best_individual = self.result[0]
            opt.batchSize = best_individual[0]
            opt.epoch = best_individual[1]
            opt.lr = best_individual[2]
            opt.activation = self.activations.index(best_individual[3])
            opt.optimizer = self.optimizer_choices.index(best_individual[4])
            return True

        # 個体と全体の速度と位置の更新
        self.update_velocities_and_positions()

        next_individual = self.population[0]
        opt.batchSize = next_individual['batchSize']
        opt.epoch = next_individual['epoch']
        opt.lr = next_individual['lr']
        opt.activation = next_individual['activation']
        opt.optimizer = next_individual['optimizer']

        return False


class MySchedulerGS:
    def __init__(self, opt):
        self.activations = ['relu', 'sigmoid', 'hardtanh', 'softmax']
        self.optimizer_choices = ['Adam', 'SGD', 'Adagrad', 'RMSprop']
        self.sum_epoch = 0
        self.min_epoch = opt.epoch_min
        self.limit_epoch = opt.epoch_limit
        self.result = []
        self.batchSize = [16, 32, 64, 128]
        self.epoch = [10, 15, 20]
        self.lr = [0.001, 0.002, 0.004, 0.01, 0.02]
        self.activation = [0, 1, 2, 3]
        self.optimizer = [0, 1, 2, 3]
        self.config = list(itertools.product(self.batchSize, self.epoch, self.lr, self.activation, self.optimizer))
        self.index = 0

    def count_epoch(self, epoch):
        if self.sum_epoch + epoch >= self.limit_epoch:
            return True
        return False

    def eval(self, epoch, history):
        if self.count_epoch(epoch):
            return True
        if epoch <= self.min_epoch:
            return False
        if history['validation_acc'][-1] < history['validation_acc'][-2]:
            return True
        else:
            return False

    def search(self, index, epoch, opt, history):
        self.sum_epoch += epoch
        self.result.append(
            [
                opt.batchSize,
                opt.epoch,
                opt.lr,
                self.activations[opt.activation],
                self.optimizer_choices[opt.optimizer],
                epoch,
                history['validation_acc'][-1],
            ]
        )
        if self.count_epoch(0) or self.index >= len(self.config):
            self.result = sorted(self.result, reverse=True, key=lambda x: x[6])
            df = pd.DataFrame(
                self.result,
                columns=[
                    'batchSize',
                    'epoch',
                    'learning rate',
                    'activation',
                    'optimizer',
                    'actual epoch',
                    'latest accuracy',
                ],
            )
            df.to_csv('./logs/optimize.txt', sep='\t')
            df.to_csv('./logs/optimize.csv', sep=',')
            opt.batchSize = self.result[0][0]
            opt.epoch = self.result[0][1]
            opt.lr = self.result[0][2]
            opt.activation = self.activations.index(self.result[0][3])
            opt.optimizer = self.optimizer_choices.index(self.result[0][4])
            return True
        opt.batchSize = self.config[self.index][0]
        opt.epoch = self.config[self.index][1]
        opt.lr = self.config[self.index][2]
        opt.activation = self.config[self.index][3]
        opt.optimizer = self.config[self.index][4]
        self.index += 1
        return False
