# -*- coding: utf-8 -*-
# @Author: Kicc
# @Date:   2018-11-24 20:51:54
# @Last Modified by:   Kicc
# @Last Modified time: 2018-12-06 12:59:18


import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn.model_selection import train_test_split
from PerformanceMeasure import PerformanceMeasure

"""
参照论文A Learning-to-Rank Approach to Software Defect Prediction
学习的模型为y=a0*x0+a1*x1+a2*x2+....+a19*x19,其中x0-x19为软件模块的20维的特征，
总共20个基因, a0,a1,a2,,,.a19.
要求出这个学习的模型，其实就是要求出a0,a1,a2,,,.a19。
然后将测试集软件模块的20维特征向量带入y=a0*x0+a1*x1+a2*x2+....+a19*x19
可求得测试集软件模块的预测缺陷个数。然后利用预测缺陷个数和真实缺陷个数对比

"""


class LTR:
    def __init__(self,
                 NP=10,
                 F=0.8,
                 CR=0.2,
                 generation=10,
                 len_x=20,
                 value_up_range=20,
                 value_down_range=-20,
                 X=None,
                 y=None):

        self.NP = NP  # 种群数量
        self.F = F  # 缩放因子
        self.CR = CR  # 交叉概率
        self.generation = generation  # 遗传代数
        self.len_x = len_x
        self.value_up_range = value_up_range
        self.value_down_range = value_down_range
        self.np_list = self.initialtion()
        self.training_data_X=X
        self.training_data_y=y

    def initialtion(self):
        # 种群初始化

        np_list = []  # 种群，染色体
        for i in range(0, self.NP):
            x_list = []  # 个体，基因
            for j in range(0, self.len_x):
                    x_list.append(self.value_down_range + random.random() *
                                  (self.value_up_range - self.value_down_range))
            np_list.append(x_list)
        return np_list

    def substract(self, a_list, b_list):
        # 列表相减
        return [a - b for (a, b) in zip(a_list, b_list)]

    def add(self, a_list, b_list):
        # 列表相加
        return [a + b for (a, b) in zip(a_list, b_list)]

    def multiply(self, a, b_list):
        # 列表的数乘
        return [a * b for b in b_list]


    def mutation(self, np_list, currentGeneration):
        """
        # 变异
        # 保证取出来的i,r1,r2,r3互不相等
        返回中间的变异种群
        """
        v_list = []
        for i in range(0, self.NP):
            r1 = random.randint(0, self.NP - 1)
            while r1 == i:
                r1 = random.randint(0, self.NP - 1)
            r2 = random.randint(0, self.NP - 1)
            while r2 == r1 | r2 == i:
                r2 = random.randint(0, self.NP - 1)
            r3 = random.randint(0, self.NP - 1)
            while r3 == r2 | r3 == r1 | r3 == i:
                r3 = random.randint(0, self.NP - 1)

            sub = self.substract(np_list[r2], np_list[r3])
            self.F *= 2 ** np.exp(1 - (self.generation /
                                       (self.generation + 1 - currentGeneration)))
            mul = self.multiply(self.F, sub)
            add = self.add(np_list[r1], mul)

            # 判断add中的基因是否符合范围, 若不符合, 就选离范围最近的那个数
            for i in range(self.len_x):
                if add[i] > self.value_up_range or add[i] < self.value_down_range:
                    add[i] = self.value_down_range + random.random() * (
                            self.value_up_range - self.value_down_range)

            v_list.append(add)

        return v_list

    def crossover(self, np_list, v_list):
        """
        np_list: 第g代初始种群
        v_list: 变异后的中间体
        """
        u_list = []
        self.CR = 0.5 * (1 + random.random())
        for i in range(0, self.NP):
            vv_list = []
            for j in range(0, self.len_x):  # len_x 是基因个数
                if (random.random() <= self.CR) or (j == random.randint(0, self.len_x - 1)):
                    vv_list.append(v_list[i][j])
                else:
                    vv_list.append(np_list[i][j])
            # 保证每个染色体至少有一个基因遗传给下一代，强制取出一个变异中间体的基因
            tmp = random.randint(0, self.len_x - 1)
            vv_list[tmp] = v_list[i][tmp]
            u_list.append(vv_list)
        return u_list

    def selection(self, u_list, np_list):
        """根据适应度函数，从初始化种群 或者 交叉后种群中选择

        """
        for i in range(0, self.NP):
            if self.Objfunction(u_list[i]) >= self.Objfunction(np_list[i]):
                np_list[i] = u_list[i]
            else:
                np_list[i] = np_list[i]
        return np_list

    def process(self):
        np_list = self.np_list
        max_x = []
        max_f = []
        for i in range(0, self.NP):
            xx = []
            xx.append(self.Objfunction(np_list[i]))
        # 将初始化的种群对应的max_f和max_xx加入
        max_f.append(max(xx))
        max_x.append(np_list[xx.index(max(xx))])

        # 迭代循环
        for i in range(0, self.generation):
            # print("iteration {0}".format(i))
            v_list = self.mutation(np_list, currentGeneration=i)  # 变异
            u_list = self.crossover(np_list, v_list)  # 杂交
            # 选择， 选择完之后的种群就是下一个迭代开始的种群
            np_list = self.selection(u_list, np_list)
            for i in range(0, self.NP):
                xx = []
                xx.append(self.Objfunction(np_list[i]))
            max_f.append(max(xx))
            max_x.append(np_list[xx.index(max(xx))])

        # 输出
        max_ff = max(max_f)
        # 用max_f.index()根据最小值max_ff找对应的染色体，说明不一定最后的染色体是最好的
        max_xx = max_x[max_f.index(max_ff)]


        print('the maximum point x =', max_xx)
        print('the maximum value y =', max_ff)
        # 画图
        x_label = np.arange(0, self.generation + 1, 1)
        plt.plot(x_label, max_f, color='blue')
        plt.xlabel('iteration')
        plt.ylabel('fx')
        plt.savefig('./iteration-f.png')
        plt.show()

        return max_xx



    def Objfunction(self, Param):
        """
        传入所有参数计算一个fpa值

        """
        pred_y = []
        for test_x in self.training_data_X:
            pred_y.append(float(np.dot(test_x, Param)))
        fpa = PerformanceMeasure(self.training_data_y, pred_y).FPA()

        return fpa

    def predict(self, testing_data_X, Param):
        pred_y = []
        for test_x in testing_data_X:
            pred_y.append(float(np.dot(test_x, Param)))

        return pred_y


if __name__ == '__main__':
    # 初始化
    # NP, F, CR, generation, len_x, value_up_range, value_down_range = initpara()
    # np_list = initialtion(NP)
    # main(np_list)
    de = SMOTEDE()
    de.process()
