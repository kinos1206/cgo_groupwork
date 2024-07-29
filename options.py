import argparse
import os
import sys
from dataclasses import dataclass

import torch


@dataclass
class TOptions:
    epoch_limit: int
    epoch_min: int
    batchSize: int
    epoch: int
    lr: float
    activation: int
    optimizer: int
    gpu_ids: list[int]
    model_dir: str
    search_method: str

    def __str__(self) -> str:
        return (
            f'epoch_limit: {self.epoch_limit}, epoch_min: {self.epoch_min}, batchSize: {self.batchSize}, '
            f'epoch: {self.epoch}, lr: {self.lr}, activation: {self.activation}, optimizer: {self.optimizer}, '
            f'gpu_ids: {self.gpu_ids}, model_dir: {self.model_dir}, search_method: {self.search_method}\n'
        )


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--epoch_limit', type=int, default=sys.maxsize, help='input limit of epoch')
        self.parser.add_argument('--epoch_min', type=int, default=5, help='input minimum number of epoch')
        self.parser.add_argument('--batchSize', type=int, default=64, help='input initial batch size')
        self.parser.add_argument('--epoch', type=int, default=20, help='input initial epoch')
        self.parser.add_argument('--lr', type=float, default=0.001, help='input initial learning rate')
        self.parser.add_argument('--activation', type=int, default=0, help='input initial activation')
        self.parser.add_argument('--optimizer', type=int, default=0, help='input initial optimizer')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--model_dir', type=str, default='./model', help='models are saved here')
        self.parser.add_argument(
            '--search_method', type=str, default='random', help='search method: [random | genetic | pso | grid]'
        )

        self.initialized = True

    def parse(self) -> TOptions:
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        parsed_opt = TOptions(
            epoch_limit=self.opt.epoch_limit,
            epoch_min=self.opt.epoch_min,
            batchSize=self.opt.batchSize,
            epoch=self.opt.epoch,
            lr=self.opt.lr,
            activation=self.opt.activation,
            optimizer=self.opt.optimizer,
            gpu_ids=self.opt.gpu_ids,
            model_dir=self.opt.model_dir,
            search_method=self.opt.search_method,
        )

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        self.mkdirs(parsed_opt.model_dir)
        file_name = os.path.join(parsed_opt.model_dir, 'option.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return parsed_opt

    def mkdir(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    def mkdirs(self, paths: list[str] | str):
        if isinstance(paths, str):
            self.mkdir(paths)
        else:
            for path in paths:
                self.mkdir(path)
