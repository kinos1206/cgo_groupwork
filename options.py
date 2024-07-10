import argparse
import os
import torch
import torch.nn.functional as f


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--epoch_limit', type=int, default=100, help='input limit of epoch')
        self.parser.add_argument('--epoch_min', type=int, default=5, help='input minimum number of epoch')
        self.parser.add_argument('--batchSize', type=int, default=64, help='input initial batch size')
        self.parser.add_argument('--epoch', type=int, default=20, help='input initial epoch')
        self.parser.add_argument('--lr', type=float, default=0.001, help='input initial learning rate')
        self.parser.add_argument('--activation', type=int, default=0, help='input initial activation')
        self.parser.add_argument('--optimizer', type=int, default=0, help='input initial optimizer')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--model_dir', type=str, default='./model', help='models are saved here')
        
        self.initialized = True

    def parse(self):
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

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        self.mkdirs(self.opt.model_dir)
        file_name = os.path.join(self.opt.model_dir, 'option.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def mkdirs(self, paths):
        if isinstance(paths, list) and not isinstance(paths, str):
            for path in paths:
                self.mkdir(path)
        else:
            self.mkdir(paths)
