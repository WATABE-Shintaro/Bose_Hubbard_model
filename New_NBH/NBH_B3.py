import copy
import random
import time
import math
import numpy as np
import argparse
import os
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch.cuda
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 変更用オプション
#################################################################################
# 　d_はデフォルトの略、コマンドからの時は変更できる
d_EPOCH = 200
d_SAMPLE_NUM = 20480
d_BATCH_SIZE = 1024
# 格子点数および粒子数
d_LATTICE = 11
d_PARTICLE = 9
# パラメータ
d_U = 2
d_J = 1
# output アウトプットするデータの名前
d_OUTPUT_FILE_NAME = "AAA"
# GPUを使う場合'cuda' cpu なら'cpu'
d_GPU = 'cuda'

################################################################################

# 学習率
LR = 0.06
LR_STEP = 50
LR_GAMMA = 0.6
MOMENTUM = 0.95
# 定数
MEMO_NAME = "memo.txt"  # 条件記録用

net_num = 20
OUTPUT_NAME = "RESULT"

PI = 3.14

# ニューラルネットワーク本体を作成
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 全結合ネットワーク
        self.l1 = nn.Linear(LATTICE, net_num)
        self.l2 = nn.Linear(net_num, 2)

    def forward(self, x):
        # reluを用いる。
        h1 = torch.tanh(self.l1(x))
        y = self.l2(h1)
        return y


class MyLoss(nn.Module):
    def __init__(self, sample_num):
        super().__init__()
        self.i_vector_array = [np.arange(LATTICE) for i in range(sample_num)]
        self.i_vector_array = np.array(self.i_vector_array)
        self.i_vector_array = torch.from_numpy(self.i_vector_array)
        self.i_vector_array = self.i_vector_array.to(DEVICE)

    def forward(self, n_vector_array, cn_array, cn_tensor_1, cn_tensor_2):
        cn_array = cn_array[:, 0] * torch.exp(cn_array[:, 1] * PI * 1j)
        cn_tensor_1 = cn_tensor_1[:, :, 0] * torch.exp(cn_tensor_1[:, :, 1] * PI * 1j)
        cn_tensor_2 = cn_tensor_2[:, :, 0] * torch.exp(cn_tensor_2[:, :, 1] * PI * 1j)
        return (-J * torch.sum(torch.sqrt(n_vector_array[:, 0:LATTICE - 1] * (n_vector_array[:, 1:LATTICE] + 1)) *
                               torch.conj(cn_tensor_1) * torch.reshape(cn_array, (-1, 1))
                               + torch.sqrt((n_vector_array[:, 0:LATTICE - 1] + 1) * n_vector_array[:, 1:LATTICE]) *
                               torch.conj(cn_tensor_2) * torch.reshape(cn_array, (-1, 1)))
                + J * torch.sum((self.i_vector_array ** 2 - (LATTICE - 1) * self.i_vector_array + (
                        LATTICE - 1) ** 2 / 4) * n_vector_array * torch.reshape(
                    torch.abs(cn_array) ** 2 ,(-1, 1)))
                + U / 2 * torch.sum((n_vector_array ** 2 - n_vector_array) * torch.reshape(
                    torch.abs(cn_array) ** 2 ,(-1, 1)))) / torch.sum(torch.abs(cn_array) ** 2)


class MyDataset(Dataset):
    def __init__(self, n_vector_array, n_vector_tensor1, n_vector_tensor2):
        super().__init__()

        self.n_vector_array = n_vector_array
        self.n_vector_tensor1 = n_vector_tensor1
        self.n_vector_tensor2 = n_vector_tensor2

    def __len__(self):
        return len(self.n_vector_array)

    def __getitem__(self, idx):
        return self.n_vector_array[idx], self.n_vector_tensor1[idx], self.n_vector_tensor2[idx]


def est_particle(n_vector_array, i, cn_array):
    cn_array = cn_array[:, 0] * torch.exp(cn_array[:, 1] * PI * 1j)
    return torch.sum(n_vector_array[:, i] * torch.abs(cn_array) ** 2) / torch.sum(
        torch.abs(cn_array) ** 2)


def montecarlo(sample_num):
    # GPUを使う場合データを変換する。to(DEVICE)
    # 返すベクトル生成
    n_vector_array = np.zeros([sample_num, LATTICE])
    for i in tqdm(range(sample_num)):
        rand_array = np.zeros([LATTICE - 1])
        j = 0
        while(j < LATTICE-1):
            n = random.randint(1, LATTICE + PARTICLE - 1)
            if not np.any(rand_array == n):
                rand_array[j] = n
                j += 1
        sorted_array = np.sort(rand_array)
        n_vector_array[i,0] = sorted_array[0] - 1
        n_vector_array[i,1:LATTICE - 1] = sorted_array[1:LATTICE - 1] - sorted_array[0:LATTICE - 2] - 1
        n_vector_array[i,LATTICE - 1] = PARTICLE + LATTICE - sorted_array[LATTICE - 2] - 1

    n_vector_array = torch.from_numpy(n_vector_array).float()
    n_vector_array = n_vector_array.to(DEVICE)
    return n_vector_array

def make_sample(n_vector_array, sample_num):
    n_vector_tensor_1 = np.zeros([sample_num, LATTICE - 1, LATTICE])
    n_vector_tensor_1 = torch.from_numpy(n_vector_tensor_1).float()
    n_vector_tensor_1 = n_vector_tensor_1.to(DEVICE)
    n_vector_tensor_2 = np.zeros([sample_num, LATTICE - 1, LATTICE])
    n_vector_tensor_2 = torch.from_numpy(n_vector_tensor_2).float()
    n_vector_tensor_2 = n_vector_tensor_2.to(DEVICE)
    for i in tqdm(range(LATTICE - 1)):
        n_vector_tensor_1[:, i] = copy.deepcopy(n_vector_array)
        n_vector_tensor_1[:, i, i] -= 1
        n_vector_tensor_1[:, i, i + 1] += 1
        n_vector_tensor_2[:, i] = copy.deepcopy(n_vector_array)
        n_vector_tensor_2[:, i, i] += 1
        n_vector_tensor_2[:, i, i + 1] -= 1

    return n_vector_tensor_1, n_vector_tensor_2


def learning():
    # オプションを表示
    print('GPU: {}'.format(GPU))
    print('# epoch: {}'.format(EPOCH))
    print('# lattice_point_num: {}'.format(LATTICE))
    print('# particle_num: {}'.format(PARTICLE))
    print('# output_file: {}'.format(OUTPUT_FILE_NAME))
    print('')

    if not os.path.exists(OUTPUT_FILE_NAME):
        os.mkdir(OUTPUT_FILE_NAME)
    # 後からわかるようにメモを出力
    with open(OUTPUT_FILE_NAME + "/" + MEMO_NAME, mode='a') as f:
        now_time = datetime.datetime.now()
        f.write("\n" + __file__ + "が実行されました。 " + now_time.strftime('%Y/%m/%d %H:%M:%S') + "\n 使用されたデータ:")
        f.write("エポック数:" + str(EPOCH) + "\n ")
        f.write("GPU:" + str(GPU) + "\n ")
        f.write("lattice_point_num:" + str(LATTICE) + "\n ")
        f.write("particle_num:" + str(PARTICLE) + "\n ")
        f.write("Lr:" + str(LR) + "\n ")
        f.write("STEP:" + str(LR_STEP) + "\n ")
        f.write("Gamma:" + str(LR_GAMMA) + "\n ")
        f.write("Momentum:" + str(MOMENTUM) + "\n ")
        f.write("sample:" + str(SAMPLE_NUM) + "\n ")
        f.write("Batch:" + str(BATCH_SIZE) + "\n ")
        f.write("U:" + str(U) + "\n ")
        f.write("J:" + str(J) + "\n ")
    print("データロード開始")

    train_n_vector_array = montecarlo(SAMPLE_NUM)
    train_n_vector_tensor_1, train_n_vector_tensor_2 = make_sample(train_n_vector_array, SAMPLE_NUM)

    test_n_vector_array = montecarlo(BATCH_SIZE)
    test_n_vector_tensor_1, test_n_vector_tensor_2 = make_sample(test_n_vector_array, BATCH_SIZE)

    train_dataset = MyDataset(train_n_vector_array, train_n_vector_tensor_1, train_n_vector_tensor_2,)
    test_dataset = MyDataset(test_n_vector_array, test_n_vector_tensor_1, test_n_vector_tensor_2,)
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

    # ニューラルネットワークを実体化
    my_net: nn.Module = MyModel()
    my_net = my_net.to(DEVICE)
    # 最適化アルゴリズム
    optimizer = optim.SGD(params=my_net.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    criterion = MyLoss(BATCH_SIZE)

    # 学習結果の保存用
    history = {'train_loss': [], 'test_loss': [], 'now_time': [], }

    # 学習開始。エポック数だけ学習を繰り返す。
    print("学習開始")
    for e in range(EPOCH):
        # 学習を行う。

        my_net.train(True)
        for tr_n_v_a, tr_n_v_t_1, tr_n_v_t_2 in train_dataloader:
            optimizer.zero_grad()
            cn_array = my_net(tr_n_v_a)
            cn_tensor_1 = my_net(tr_n_v_t_1)
            cn_tensor_2 = my_net(tr_n_v_t_2)
            energy = criterion(tr_n_v_a, cn_array, cn_tensor_1, cn_tensor_2)
            energy.backward()
            optimizer.step()
        train_loss = energy.item()
        scheduler.step()
        # テストを行う。

        history['train_loss'].append(train_loss)
        my_net.eval()
        with torch.no_grad():
            for te_n_v_a, te_n_v_t_1, te_n_v_t_2 in test_dataloader:
                cn_array = my_net(te_n_v_a)
                cn_tensor_1 = my_net(te_n_v_t_1)
                cn_tensor_2 = my_net(te_n_v_t_2)
                energy = criterion(te_n_v_a, cn_array, cn_tensor_1, cn_tensor_2)
        test_loss = energy.item()
        history['test_loss'].append(test_loss)
        # 経過を記録して、表示。
        now_time = datetime.datetime.now()
        history['now_time'].append(now_time.strftime('%Y/%m/%d %H:%M:%S'))
        print('Train Epoch: {}/{} \t TrainLoss: {:.6f} \t TestLoss: {:.6f} \t time: {} \t lr:{}'
              .format(e + 1, EPOCH, train_loss, test_loss, now_time.strftime('%Y/%m/%d %H:%M:%S'),
                      scheduler.get_last_lr()[0]))

    print("学習終了")
    # 予測を行う。
    my_net.eval()
    n_vector_result = np.zeros([LATTICE])
    with torch.no_grad():
        criterion = MyLoss(SAMPLE_NUM)
        est_n_vector_array = montecarlo(SAMPLE_NUM)
        est_n_vector_tensor_1, est_n_vector_tensor_2 = make_sample(est_n_vector_array, SAMPLE_NUM)
        cn_array = my_net(est_n_vector_array)
        cn_tensor_1 = my_net(est_n_vector_tensor_1)
        cn_tensor_2 = my_net(est_n_vector_tensor_2)
        energy = criterion(est_n_vector_array, cn_array, cn_tensor_1, cn_tensor_2)
        for i in range(LATTICE):
            n_vector_result[i] = est_particle(est_n_vector_array, i, cn_array)
        est_loss = energy.item()

    # 結果をセーブする。
    if not os.path.exists(args.out + "/" + OUTPUT_NAME):
        os.mkdir(args.out + "/" + OUTPUT_NAME)
    torch.save(my_net.state_dict(), args.out + "/" + OUTPUT_NAME + "/" + 'model.pth')
    with open(args.out + "/" + OUTPUT_NAME + "/" + 'result.txt', mode='a') as f:
        f.write("\n \n " + str(est_loss))
        f.write("\n \n " + np.array2string(n_vector_result, separator=','))
    with open(args.out + "/" + OUTPUT_NAME + "/" + 'history.txt', mode='a') as f:
        f.write("\n \n " + str(history))

    print("終了")
    # 終了時間メモ
    now_time = datetime.datetime.now()
    with open(args.out + "/" + MEMO_NAME, mode='a') as f:
        f.write("\n終了しました" + now_time.strftime('%Y/%m/%d %H:%M:%S') + "\n\n")

    return str(est_loss), str(n_vector_result)


"""
def forward(n_vector_array, cn_array, cn_tensor_1, cn_tensor_2, n_correction_array):
    i_vector_array = [np.arange(LATTICE) for i in range(SAMPLE_NUM * 10)]
    i_vector_array = np.array(i_vector_array)
    i_vector_array = torch.from_numpy(i_vector_array)
    i_vector_array = i_vector_array.to(DEVICE)
    cn_array = cn_array[:, 0] * torch.exp(cn_array[:, 1] * 1j)
    cn_tensor_1 = cn_tensor_1[:, :, 0] * torch.exp(cn_tensor_1[:, :, 1] * 1j)
    cn_tensor_2 = cn_tensor_2[:, :, 0] * torch.exp(cn_tensor_2[:, :, 1] * 1j)
    d = torch.sum(torch.abs(cn_array) ** 2)
    q = torch.reshape(n_correction_array, (-1, 1))
    a = -J * torch.sum(torch.sqrt(n_vector_array[:, 0:LATTICE - 1] * (n_vector_array[:, 1:LATTICE] + 1)) *
                       torch.t(torch.conj(cn_tensor_1) * cn_array)
                       + torch.sqrt((n_vector_array[:, 0:LATTICE - 1] + 1) * n_vector_array[:, 1:LATTICE]) *
                       torch.t(torch.conj(cn_tensor_2) * cn_array) * torch.reshape(n_correction_array,
                                                                                   (-1, 1))) / d
    c0 = n_vector_array ** 2
    c1 = n_vector_array
    b = J * torch.sum(
        (i_vector_array ** 2 - (LATTICE - 1) * i_vector_array + (
                LATTICE - 1) ** 2 / 4) * n_vector_array * torch.reshape(torch.abs(cn_array) ** 2,
                                                                        (-1, 1)) * torch.reshape(n_correction_array,
                                                                                                 (-1, 1))) / d
    c = U / 2 * torch.sum(
        (n_vector_array ** 2 - n_vector_array) * torch.reshape(torch.abs(cn_array) ** 2, (-1, 1)) * torch.reshape(
            n_correction_array,
            (-1, 1))) / d

    return a + b + c
"""

if __name__ == '__main__':
    # コマンドラインからプログラムを動かす時のオプションを実装
    parser = argparse.ArgumentParser(description='Pytorch' + __file__)
    parser.add_argument('--epoch', '-e', type=int, default=d_EPOCH,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batch', '-b', type=int, default=d_BATCH_SIZE,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=str, default=d_GPU,
                        help='if you want to use GPU, select cuda. cpu for cpu')
    parser.add_argument('--M', '-m', default=d_LATTICE,
                        help='number of lattice points')
    parser.add_argument('--N', '-n', default=d_PARTICLE,
                        help='number of particle')
    parser.add_argument('--U', '-u', default=d_U,
                        help='value of U')
    parser.add_argument('--J', '-j', default=d_J,
                        help='value of J')
    parser.add_argument('--out', '-o', default=d_OUTPUT_FILE_NAME,
                        help='output file name')
    parser.add_argument('--sample', '-s', default=d_SAMPLE_NUM,
                        help='output file name')
    args = parser.parse_args()

    EPOCH = args.epoch
    BATCH_SIZE = args.batch
    GPU = args.gpu
    LATTICE = args.M
    PARTICLE = args.N
    U = args.U
    J = args.J
    OUTPUT_FILE_NAME = args.out
    SAMPLE_NUM = args.sample
    DEVICE = torch.device(GPU)

    result = learning()
    print("ene" + result[0])
    print("num" + result[1])
    """
    aaa = MyModel()
    aaa.to(DEVICE)
    ddd = MyLoss(SAMPLE_NUM)
    ggg = optim.SGD(params=aaa.parameters(), lr=LR, momentum=MOMENTUM)
    start = time.time()
    bb1 = montecarlo(SAMPLE_NUM)
    print("monte:{0}".format(time.time() - start) + "[sec]")
    start = time.time()
    bb2, bb3 = make_sample(bb1, SAMPLE_NUM)
    print("make:{0}".format(time.time() - start) + "[sec]")
    start = time.time()
    bbb = montecarlo_correction(bb1, SAMPLE_NUM)
    print("corr:{0}".format(time.time() - start) + "[sec]")
    start = time.time()
    cc1 = aaa(bb1)
    cc2 = aaa(bb2)
    cc3 = aaa(bb3)
    print("net:{0}".format(time.time() - start) + "[sec]")
    start = time.time()
    fff = ddd(bb1, cc1, cc2, cc3, bbb)
    print("loss:{0}".format(time.time() - start) + "[sec]")
    start = time.time()
    fff.backward()
    print("backward:{0}".format(time.time() - start) + "[sec]")
    start = time.time()
    ggg.step()
    print("optim:{0}".format(time.time() - start) + "[sec]")
    print(fff)
"""
