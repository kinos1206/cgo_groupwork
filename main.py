import os
import torch
import torch.nn.functional as f
from matplotlib import pyplot as plt
from dataloader import load_MNIST
from Network import MyNet
from options import Options
from tune import MyScheduler, , MySchedulerGA 

#引数の読み込み
opt = Options().parse()

#ハイパーパラメータスケジューラの読み込み
if opt.search_method == 'genetic':
    scheduler = MySchedulerGA(opt)
else:
    scheduler = MyScheduler(opt)

#グラフの用意
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

loop = 1

while True:
    #学習結果の保存
    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_acc": []
    }

    #データのロード
    data_loader = load_MNIST(batch=opt.batchSize)

    #ネットワーク構造の構築
    net = MyNet(opt, scheduler.activations).cuda(opt.gpu_ids[0])
    print(net)

    #最適化方法の設定
    optimizer_choices = scheduler.optimizer_choices
    optimizer = eval('torch.optim.' + optimizer_choices[opt.optimizer])
    optimizer = optimizer(params=net.parameters(), lr=opt.lr)

    for e in range(opt.epoch):
        """ 学習部分 """
        loss = None
        train_loss = 0.0
        net.train() #学習モード
        print("\nTrain start")
        for i,(data,target) in enumerate(data_loader["train"]):
            data,target = data.cuda(opt.gpu_ids[0]),target.cuda(opt.gpu_ids[0])

            #勾配の初期化
            optimizer.zero_grad()
            #順伝搬 -> 逆伝搬 -> 最適化
            output = net(data)
            loss = f.nll_loss(output,target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % 100 == 99:
                print("Training: {} epoch. {} iteration. Loss:{}".format(e+1,i+1,loss.item()))

        train_loss /= len(data_loader["train"])
        print("Training loss (ave.): {}".format(train_loss))
        history["train_loss"].append(train_loss)

        """検証部分"""
        print("\nValidation start")
        net.eval() #検証モード(Validation)
        val_loss = 0.0
        accuracy = 0.0

        with torch.no_grad():
            for data,target in data_loader["validation"]:
                data,target = data.cuda(opt.gpu_ids[0]),target.cuda(opt.gpu_ids[0])

                #順伝搬の計算
                output = net(data)
                loss = f.nll_loss(output,target).item()
                val_loss += f.nll_loss(output,target,reduction='sum').item()
                predict = output.argmax(dim=1,keepdim=True)
                accuracy += predict.eq(target.view_as(predict)).sum().item()

        val_loss /= len(data_loader["validation"].dataset)
        accuracy /= len(data_loader["validation"].dataset)

        print("Validation loss: {}, Accuracy: {}\n".format(val_loss,accuracy))
        history["validation_loss"].append(val_loss)
        history["validation_acc"].append(accuracy)

        if loop > 0:
            #epoch数の表示
            print('total epoch: ' + str(scheduler.sum_epoch + e + 1))
            #学習を続けるか評価
            if scheduler.eval(e+1, history):
                break
    
    if loop == -1: break

    #ハイパーパラメータの更新, 最適化を続けるか判定
    if scheduler.search(loop-1, e+1, opt, history):
        #最も
        loop = -1
        continue

    loop += 1


#モデルを保存する
os.makedirs('./model', exist_ok=True)
PATH = "./model/my_mnist_model_%d_%d_%f_%d_%d.pt" % (opt.batchSize, opt.epoch, opt.lr, opt.activation, opt.optimizer)
torch.save(net.state_dict(), PATH)

#グラフをプロットする
ax1.plot(history["train_loss"], linestyle='-', label='train_loss_%d_%d_%f_%d_%d' % (opt.batchSize, opt.epoch, opt.lr, opt.activation, opt.optimizer))
ax1.plot(history["validation_loss"], linestyle='-', label='validation_loss_%d_%d_%f_%d_%d' % (opt.batchSize, opt.epoch, opt.lr, opt.activation, opt.optimizer))
ax2.plot(history["validation_acc"], linestyle='-', label='validation_acc_%d_%d_%f_%d_%d' % (opt.batchSize, opt.epoch, opt.lr, opt.activation, opt.optimizer))
ax1.legend()
ax2.legend()
dirname = "./logs/"
os.makedirs(dirname, exist_ok=True)
fig_loss.savefig(dirname + "train_loss.png")
fig_acc.savefig(dirname + "test_acc.png")
