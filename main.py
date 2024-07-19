import os
import time
from typing import Sized

import pandas as pd
import torch
import torch.nn.functional as f
from matplotlib import pyplot as plt

from dataloader import load_MNIST
from Network import MyNet
from options import Options
from tune import MyScheduler, MySchedulerGA, MySchedulerGS, MySchedulerPSO

# 引数の読み込み
opt = Options().parse()

# ハイパーパラメータスケジューラの読み込み
if opt.search_method == 'genetic':
    scheduler = MySchedulerGA(opt)
elif opt.search_method == 'pso':
    scheduler = MySchedulerPSO(opt)
elif opt.search_method == 'grid':
    scheduler = MySchedulerGS(opt)
else:
    scheduler = MyScheduler(opt)

# グラフの用意
fig_loss, ax1 = plt.subplots()
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_title('Training loss')
ax1.grid()
fig_acc, ax2 = plt.subplots()
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.set_title('Test accuracy')
ax2.grid()

# 結果保存用ディレクトリの作成
os.makedirs('./logs', exist_ok=True)  # ディレクトリを変更
os.makedirs('./logs/history', exist_ok=True)

loop = 1

start_time = time.time()  # 全体、最適化部分の時間計測の開始

while True:
    # 学習結果の保存
    history = {'train_loss': [], 'validation_loss': [], 'validation_acc': []}

    # データのロード
    data_loader = load_MNIST(batch=opt.batchSize)

    # ネットワーク構造の構築
    net = MyNet(opt, scheduler.activations).cuda(opt.gpu_ids[0])
    print(net)

    # 最適化方法の設定
    optimizer_choices = scheduler.optimizer_choices
    optimizer = eval('torch.optim.' + optimizer_choices[opt.optimizer])
    optimizer = optimizer(params=net.parameters(), lr=opt.lr)

    for e in range(opt.epoch):
        """ 学習部分 """
        loss = None
        train_loss = 0.0
        net.train()  # 学習モード
        print('\nTrain start')
        for i, (data, target) in enumerate(data_loader['train']):
            data, target = data.cuda(opt.gpu_ids[0]), target.cuda(opt.gpu_ids[0])

            # 勾配の初期化
            optimizer.zero_grad()
            # 順伝搬 -> 逆伝搬 -> 最適化
            output = net(data)
            loss = f.nll_loss(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % 100 == 99:
                print(f'Training: {e+1} epoch. {i+1} iteration. Loss:{loss.item()}')

        train_loss /= len(data_loader['train'])
        print('Training loss (ave.): {}'.format(train_loss))
        history['train_loss'].append(train_loss)

        """ 検証部分 """
        print('\nValidation start')
        net.eval()  # 検証モード(Validation)
        val_loss = 0.0
        accuracy = 0.0

        with torch.no_grad():
            for data, target in data_loader['validation']:
                data, target = data.cuda(opt.gpu_ids[0]), target.cuda(opt.gpu_ids[0])

                # 順伝搬の計算
                output = net(data)
                loss = f.nll_loss(output, target).item()
                val_loss += f.nll_loss(output, target, reduction='sum').item()
                predict = output.argmax(dim=1, keepdim=True)
                accuracy += predict.eq(target.view_as(predict)).sum().item()

        dataset = data_loader['validation'].dataset
        if isinstance(dataset, Sized):
            val_loss /= len(dataset)
            accuracy /= len(dataset)
        else:
            raise TypeError('Dataset does not support len()')

        print(f'Validation loss: {val_loss}, Accuracy: {accuracy}\n')
        history['validation_loss'].append(val_loss)
        history['validation_acc'].append(accuracy)

        if loop > 0:
            # epoch数の表示
            print(f'total epoch: {scheduler.sum_epoch + e + 1}')
            # 学習を続けるか評価
            if scheduler.eval(e + 1, history):
                break

    if loop == -1:
        break

    # 学習履歴の保存
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'./logs/history/history_{loop - 1}.csv', index=False)  # ログディレクトリに保存

    # ハイパーパラメータの更新, 最適化を続けるか判定
    if scheduler.search(loop - 1, e + 1, opt, history):
        end_sheduler = time.time()  # 最適化部分の時間計測の終了
        print('\n--------- Train with optimised hyper parameters ---------\n')
        loop = -1
        continue

    loop += 1

end_time = time.time()  # 全体の時間計測の終了
elapsed_time = end_time - start_time
time_sheduler = end_sheduler - start_time

# モデルを保存する
os.makedirs(f'{opt.model_dir}', exist_ok=True)
PATH = './model/my_mnist_model_%d_%d_%f_%d_%d.pt' % (opt.batchSize, opt.epoch, opt.lr, opt.activation, opt.optimizer)
torch.save(net.state_dict(), PATH)

# グラフをプロット、保存する
ax1.plot(
    history['train_loss'],
    linestyle='-',
    label='train_loss_%d_%d_%f_%d_%d' % (opt.batchSize, opt.epoch, opt.lr, opt.activation, opt.optimizer),
)
ax1.plot(
    history['validation_loss'],
    linestyle='-',
    label='validation_loss_%d_%d_%f_%d_%d' % (opt.batchSize, opt.epoch, opt.lr, opt.activation, opt.optimizer),
)
ax2.plot(
    history['validation_acc'],
    linestyle='-',
    label='validation_acc_%d_%d_%f_%d_%d' % (opt.batchSize, opt.epoch, opt.lr, opt.activation, opt.optimizer),
)
ax1.legend()
ax2.legend()
fig_loss.savefig('./logs/train_loss.png')
fig_acc.savefig('./logs/test_acc.png')

# 結果を保存する
message = ''
message += f'Total epochs: {scheduler.sum_epoch}\n'
message += f'Elapsed time (total): {elapsed_time:.2f} seconds\n'
message += f'Elapsed time (optimization): {time_sheduler:.6f}\n'
with open('./logs/optimization_summary.txt', 'w') as f:
    f.write(f'{message}')
