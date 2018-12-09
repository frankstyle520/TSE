# -*- coding: utf-8 -*-
# @Author: Kicc
# @Date:   2018-11-24 20:51:54
# @Last Modified by:   Kicc
# @Last Modified time: 2018-12-06 12:59:18


import numpy as np
import matplotlib.pyplot as plt
import math
import random
from Smote import Smote
from sklearn.model_selection import train_test_split
from PerformanceMeasure import PerformanceMeasure

"""
总共三个基因, ratio, k, r.
ratio, k都是离散的
r是连续的变量
        
            ratio: [0.5, 1.0, 2.0, 4.0]
            k: [1-20], int
            r: [0.1, 5], float
        
"""


class SMOTEDE:
    def __init__(self,
                 NP=10,
                 F=0.7,
                 CR=0.3,
                 generation=10,
                 len_x=10,
                 kRange=list(range(1, 21)),
                 ratiovalue_up_range=1.5,
                 ratiovalue_down_range=0.5,
                 rvalue_up_range=5.0,
                 rvalue_down_range=0.1,
                 X=None,
                 y=None,
                 classifer=None):

        self.NP = NP   # 种群数量
        self.F = F   # 缩放因子
        self.CR = CR   # 交叉概率
        self.generation = generation   # 遗传代数
        self.len_x = len_x
        self.ratiovalue_up_range = ratiovalue_up_range
        self.ratiovalue_down_range = ratiovalue_down_range
        self.rvalue_up_range = rvalue_up_range
        self.rvalue_down_range = rvalue_down_range
        self.kRange = kRange
        self.classifer=classifer

        self.np_list = self.initialtion()

        self.trainX, self.trainy, self.validX, self.validy = dataProcess(
            X=X, y=y).generateData()

    def initialtion(self):
        # 种群初始化

        np_list = []   # 种群，染色体
        for i in range(0, self.NP):
            x_list = []   # 个体，基因
            for j in range(0, self.len_x):
                # cuz len_x==3, and the third one is r.
                if j == 2:
                    x_list.append(self.rvalue_down_range + random.random() *
                                  (self.rvalue_up_range - self.rvalue_down_range))
                elif j == 0:
                    # ratio
                    x_list.append(self.ratiovalue_down_range + random.random() *
                                  (self.ratiovalue_up_range - self.ratiovalue_down_range))
                elif j == 1:
                    # k
                    x_list.append(random.choice(self.kRange))
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

    def getNeareast(self, value, ranges):
        """ ratio&k是离散的，变异过程中可能取到不在范围内的值
            进行微调

        """
        res = ranges[0]
        minimum = abs(value - ranges[0])
        for r in ranges:
            if minimum > abs(value - r):
                minimum = abs(value - r)
                res = r
        return res

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
            self.F *= 2**np.exp(1 - (self.generation /
                                     (self.generation + 1 - currentGeneration)))
            mul = self.multiply(self.F, sub)
            add = self.add(np_list[r1], mul)

            # 判断add中的基因是否符合范围, 若不符合, 就选离范围最近的那个数
            if add[0] >self.ratiovalue_up_range or add[0]<self.ratiovalue_down_range:
                add[0]=self.ratiovalue_down_range + random.random() * (self.ratiovalue_up_range - self.ratiovalue_down_range)
            if add[1] not in self.kRange:
                add[1] = self.getNeareast(add[1], self.kRange)
            if add[2] >self.rvalue_up_range or add[2]<self.rvalue_down_range:
                add[2]=self.rvalue_down_range + random.random() * (self.rvalue_up_range - self.rvalue_down_range)
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
            if self.smoteObj(u_list[i]) >= self.smoteObj(np_list[i]):
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
            xx.append(self.smoteObj(np_list[i]))
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
                xx.append(self.smoteObj(np_list[i]))
            max_f.append(max(xx))
            max_x.append(np_list[xx.index(max(xx))])

        # 输出
        max_ff = max(max_f)
        # 用max_f.index()根据最小值max_ff找对应的染色体，说明不一定最后的染色体是最好的
        max_xx = max_x[max_f.index(max_ff)]

        '''
        print('the maximum point x =', max_xx)
        print('the maximum value y =', max_ff)
        # 画图
        x_label = np.arange(0, self.generation + 1, 1)
        plt.plot(x_label, max_f, color='blue')
        plt.xlabel('iteration')
        plt.ylabel('fx')
        plt.savefig('./iteration-f.png')
        plt.show()
        '''
        return max_xx

    def object_function(self, x):
        """
        适应度函数注册
        x是一个list
        x^2 - (10*cos(2*pi*x)+10)
        """
        f = 0
        for i in range(0, len(x)):
            f = f + (x[i] ** 2 - (10 * math.cos(2 * np.pi * x[i])) + 10)
        return f

    def smoteObj(self, smoteParam):
        """传入所有参数计算一个fpa值
        smoteParam: [ratio, k, r]
        param: ratio: smote的比例
               k: 最邻近个数
               p: minkowski 指标

        """
        def getFPA(model, smote_X, smote_y, validX, validy):
            trainmodel = model.fit(smote_X, smote_y)
            model_pred_y = np.around(trainmodel.predict(validX))
            model_fpa = PerformanceMeasure(validy, model_pred_y).FPA()
            return model_fpa

        ratio = smoteParam[0]
        k = smoteParam[1]
        r = smoteParam[2]
        smote_X, smote_y = Smote(
            X=self.trainX, Y=self.trainy, ratio=ratio, k=k, r=r).over_sampling()

        # get the FPA with bbr model.
        fpa = getFPA(model=self.classifer, smote_X=smote_X, smote_y=smote_y,
                     validX=self.validX, validy=self.validy)

        return fpa


class dataProcess:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def generateData(self):
        """将数据集分成4份, 3份训练集, 1份验证集

        """

        trainX, validX, trainy, validy = train_test_split(
            self.X, self.y, test_size=0.25, random_state=0)

        return trainX, trainy, validX, validy


if __name__ == '__main__':
    # 初始化
    # NP, F, CR, generation, len_x, value_up_range, value_down_range = initpara()
    # np_list = initialtion(NP)
    # main(np_list)
    de = SMOTEDE()
    de.process()
