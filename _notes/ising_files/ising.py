import numpy as np
import matplotlib.pyplot as plt
from numba import jit
def initial_disorder_spin_field(M,N):
    return np.random.choice([-1,1],size=(M,N))
def initial_order_spin_field(M,N):
    return np.ones((M,N))


# 计算能量差
@jit(nopython=True)
def delta_energy(spin_field, i, j, M, N, H=0, J=1):
    '''
    spin_field:自旋场
    H:外磁场
    J:自旋相互作用
    '''
    delta_E = 2 * J * spin_field[i, j] * (spin_field[(i - 1) % M, j] + spin_field[(i + 1) % M, j]
                                          + spin_field[i, (j - 1) % N] + spin_field[i, (j + 1) % N]) + 2 * spin_field[i, j] * H
    return delta_E
# 单粒子翻转算法，如果能量更小直接翻转，否则具有一定概率翻转
@jit(nopython=True)
def metropolis_update(spin_field, T, k=1):
    '''
    T:温度
    k玻尔兹曼常数
    '''
    M, N = spin_field.shape
    i, j = np.random.randint(0, N), np.random.randint(0, M)
    delta_E = delta_energy(spin_field, i, j, M, N, H=0, J=1)

    if delta_E <= 0:
        spin_field[i, j] = -spin_field[i, j]
    elif np.exp((-delta_E) / (k * (T))) > np.random.rand():
        spin_field[i, j] = -spin_field[i, j]
    return spin_field
# cluster算法，一次性可以更新很多，类似于广度搜索算法，我们选取一个点，其实格点和自旋相同的就会有一定可能性放到一个cluster中，继续对新的格点进行操作，
# 一直到cluster不再增长，把周围的和本身一样的大部分挑出来一起旋转可以更快收敛
@jit(nopython=True)
def wolff_update(spin_field, T, J=1, k=1):
    '''
    T:温度
    k玻尔兹曼常数
    '''
    M, N = spin_field.shape
    i, j = np.random.randint(0, N), np.random.randint(0, M)
    P_add = 1 - np.exp(-2 * J / (k * (T)))
    stack = [(i, j)]
    cluster = {(i, j)}
    spin = spin_field[i, j]
    while stack:
        i, j = stack.pop()
        neighbors = [(i, (j + 1) % N), (i, (j - 1) % N), ((i + 1) % M, j), ((i - 1) % M, j)]
        for x_, y_ in neighbors:
            if spin_field[x_, y_] == spin and (x_, y_) not in cluster and np.random.rand() <= P_add:
                stack.append((x_, y_))
                cluster.add((x_, y_))
    for i, j in cluster:
        spin_field[i, j] = -spin_field[i, j]
    return spin_field



# 模拟形成最后的自旋场，可以根据这个自旋场计算各种物理量
def simulate_spin_field(spin_field, method, T, steps):
    '''
    Method:
    Metropolis单粒子翻转算法
    Wolff团簇翻转算法
    '''
    if method == 'Metropolis':
        for i in range(steps):
            spin_field = metropolis_update(spin_field, T)
    if method == 'Wolff':
        for i in range(steps):
            spin_field = wolff_update(spin_field, T)
    return spin_field
# 开始使用训练好的模型计算一些物理量，首先是能量和磁场大小
def average_magnetization(spin_field):
    # 由于我们关注的是大小，正负其实是等价的所以
    aver_magnetization = np.abs(np.mean(spin_field))
    return aver_magnetization

def total_energy(spin_field):
    energy = 0
    N, M = spin_field.shape
    for i in range(N):
        for j in range(M):
            # 计算能量，并且使用%N和%M给出周期性边界条件，可以模拟无限大体系，避免有限尺寸效应
            energy += -spin_field[i, j] * (spin_field[(i + 1) % N, j] + spin_field[i, (j + 1) % M])
    return energy


# 给出磁化率和比热的计算
@jit(nopython=True)
def specific_heat(spin_field, T, k=1):
    energy_sum = 0
    energy_square_sum = 0
    M, N = spin_field.shape

    for i in range(M):
        for j in range(N):
            delta_E = delta_energy(spin_field, i, j, M, N)
            energy_sum += delta_E
            energy_square_sum += delta_E ** 2
    average_energy = energy_sum / (M * N)
    average_energy_square = energy_square_sum / (M * N)
    specific_heat = (average_energy_square - average_energy ** 2) / (k * T ** 2)
    return specific_heat


@jit(nopython=True)
def susceptibility(spin_field, T, k=1):
    magnetization_sum = 0
    magnetization_square_sum = 0
    M, N = spin_field.shape

    for i in range(M):
        for j in range(N):
            magnetization = spin_field[i, j]
            magnetization_sum += magnetization
            magnetization_square_sum += magnetization ** 2

    average_magnetization = magnetization_sum / (M * N)
    average_magnetization_square = magnetization_square_sum / (M * N)
    susceptibility = (average_magnetization_square - average_magnetization ** 2) / (k * T)

    return susceptibility