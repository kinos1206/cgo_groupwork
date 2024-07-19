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
        self.max_no_improvement = 3
        # 山登り法でのノイズの導入確率
        self.noise_probability = 1.0  # ノイズを加える確率(今回は必ずノイズを加える)
        # 現在最適化しているパラメータ
        self.current_target: HyperParam = HyperParam.BATCH_SIZE
        print(f'Current target: {self.current_target.name}')
        self.param_induces = [
            self.batchSize.index(opt.batchSize),
            self.epoch.index(opt.epoch),
            self.lr.index(opt.lr),
            opt.activation,
            opt.optimizer,
        ]

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

        # 最良のaccuracyが更新されなかった回数が上限に達した場合
        is_best = False
        if self.no_improvement_count >= self.max_no_improvement:
            try:
                # 次のパラメータに移行
                self.current_target = HyperParam(self.current_target + 1)
                print(f'Change target to {self.current_target.name}')
            except ValueError:
                is_best = True

        # 終了条件
        if self.count_epoch(0) or is_best:
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
            opt.batchSize = self.best_conf.batchSize
            opt.epoch = self.best_conf.epoch
            opt.lr = self.best_conf.lr
            opt.activation = self.best_conf.activation
            opt.optimizer = self.best_conf.optimizer
            return True

        # 次の設定を生成
        if self.current_target == HyperParam.BATCH_SIZE:
            opt.batchSize = self.get_neighbor(self.batchSize)
        elif self.current_target == HyperParam.EPOCH:
            opt.epoch = self.get_neighbor(self.epoch)
        elif self.current_target == HyperParam.LR:
            opt.lr = self.get_neighbor(self.lr)
        elif self.current_target == HyperParam.ACTIVATION:
            opt.activation = self.get_neighbor(self.activation)
        elif self.current_target == HyperParam.OPTIMIZER:
            opt.optimizer = self.get_neighbor(self.optimizer)

        return False

    @overload
    def get_neighbor(self, choices: Sequence[int]) -> int: ...

    @overload
    def get_neighbor(self, choices: Sequence[float]) -> float: ...

    def get_neighbor(self, choices: Sequence[int | float]) -> int | float:
        if random.random() < self.noise_probability:
            current_index = self.param_induces[self.current_target]
            if current_index == 0:
                return choices[1]
            elif current_index == len(choices) - 1:
                return choices[-2]
            else:
                return random.choice([choices[current_index - 1], choices[current_index + 1]])
        return choices[self.param_induces[self.current_target]]

