import itertools
import numpy as np
import test
import time

aaa = list(itertools.combinations_with_replacement(range(2), 3))

print(aaa)


def hhh(n, m):
    return bbb(n, 1, m)


def bbb(n, nt, mt):
    if nt < n:
        ddd = bbb(n, nt + 1, mt)
        fff = np.full((len(ddd), 1), 0,dtype=np.int64)
        ccc = np.hstack([ddd, fff])
        for i in range(1, mt + 1):
            ddd = bbb(n, nt + 1, mt - i)
            fff = np.full((len(ddd), 1), i,dtype=np.int64)
            eee = np.hstack([ddd, fff])
            ccc = np.vstack([ccc, eee])
        return ccc
    else:
        return np.full((1, 1), mt,dtype=np.int64)


# 時間計測開始
time_sta = time.perf_counter()
# 処理を書く（ここでは5秒停止する）
print("\n")
iii = hhh(11, 12)
print(iii[1])
# 時間計測終了
time_end = time.perf_counter()
# 経過時間（秒）
tim = time_end- time_sta
print(tim)
#[結果] 4.99997752884417


# 時間計測開始
time_sta = time.perf_counter()
# 処理を書く（ここでは5秒停止する）
print("\n")
qqq = test.test()
iii = qqq.hhh(11,12)
print(iii[1])
# 時間計測終了
time_end = time.perf_counter()
# 経過時間（秒）
tim = time_end- time_sta
print(tim)
#[結果] 4.99997752884417


