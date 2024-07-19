import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence, overload

import pandas as pd

from options import TOptions
from tune import MyScheduler

@dataclass
class THyperParams:
    batchSize: int
    epoch: int
    lr: float
    activation: int
    optimizer: int


class HyperParam(IntEnum):
    BATCH_SIZE = 0
    EPOCH = 1
    LR = 2
    ACTIVATION = 3
    OPTIMIZER = 4


# 山登り法: 最適解の近傍を探索する
class HillClimbScheduler(MyScheduler):
    def __init__(self, opt: TOptions) -> None:
        super().__init__(opt)
        # self.batchSize = [16, 32, 64, 128]
        # self.epoch = [10, 15, 20]
        # self.lr = [0.001, 0.002, 0.004, 0.01, 0.02]
        # self.activation = [0, 1, 2, 3]
        # self.optimizer = [0, 1, 2, 3]

        # 最良のaccuracyとそのときの設定を保持
        self.best_conf = THyperParams(opt.batchSize, opt.epoch, opt.lr, opt.activation, opt.optimizer)
        self.best_acc = float('-inf')
        # 最良のaccuracyが更新されなかった回数
        self.no_improvement_count = 0
        # 最良のaccuracyが更新されなかった回数の上限（終了条件）
        self.max_no_improvement = 5
        # 山登り法でのノイズの導入確率
        self.noise_probability = 1.0  # ノイズを加える確率(今回は必ずノイズを加える)
        # 現在の設定のインデックス
        self.param_induces = [
            self.batchSize.index(opt.batchSize),
            self.epoch.index(opt.epoch),
            self.lr.index(opt.lr),
            opt.activation,
            opt.optimizer,
        ]
        # 前回の設定のインデックス
        self.prev_param_induces = [0, 0, 0, 0, 0]

    def search(self, index: int, epoch: int, opt: TOptions, history: dict) -> bool:
        self.sum_epoch += epoch

        # 現在の設定を評価
        current_acc = history['validation_acc'][-1]
        current_conf = THyperParams(opt.batchSize, opt.epoch, opt.lr, opt.activation, opt.optimizer)

        if current_acc > self.best_acc:
            self.best_conf = current_conf
            self.best_acc = current_acc
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            self.param_induces = [*self.prev_param_induces]

        self.result.append(
            [
                opt.batchSize,
                opt.epoch,
                opt.lr,
                self.activations[opt.activation],
                self.optimizer_choices[opt.optimizer],
                epoch,
                current_acc,
            ]
        )

        # 終了条件
        if self.count_epoch(0) or self.no_improvement_count >= self.max_no_improvement:
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
            df.to_csv(f'{self.dirname}/optimize.txt', sep='\t')
            df.to_csv(f'{self.dirname}/optimize.csv', sep=',')
            opt.batchSize = self.best_conf.batchSize
            opt.epoch = self.best_conf.epoch
            opt.lr = self.best_conf.lr
            opt.activation = self.best_conf.activation
            opt.optimizer = self.best_conf.optimizer
            return True

        # 一度にすべてのパラメータを更新
        opt.batchSize = self.get_neighbor(self.batchSize, HyperParam.BATCH_SIZE)
        opt.epoch = self.get_neighbor(self.epoch, HyperParam.EPOCH)
        opt.lr = self.get_neighbor(self.lr, HyperParam.LR)
        opt.activation = self.get_neighbor(self.activation, HyperParam.ACTIVATION)
        opt.optimizer = self.get_neighbor(self.optimizer, HyperParam.OPTIMIZER)
        self.prev_param_induces = [*self.param_induces]

        return False

    @overload
    def get_neighbor(self, choices: Sequence[int], target_idx: int) -> int: ...

    @overload
    def get_neighbor(self, choices: Sequence[float], target_idx: int) -> float: ...

    def get_neighbor(self, choices: Sequence[int | float], target_idx: int) -> int | float:
        if random.random() < self.noise_probability:
            current_index = self.param_induces[target_idx]
            if current_index == 0:
                self.param_induces[target_idx] = 1
            elif current_index == len(choices) - 1:
                self.param_induces[target_idx] = len(choices) - 2
            else:
                self.param_induces[target_idx] = random.choice([current_index - 1, current_index + 1])
        return choices[self.param_induces[target_idx]]

