# -*- coding: utf-8 -*-
# @Author: yuxiao
# @Date:   2019-1-20 14:42:05
# @Last Modified by:   yuxiao
# @Last Modified time: 2019-1-20 14:42:05

from sklearn.neighbors import NearestNeighbors
import numpy as np
import random


class Oversampling:
    def __init__(self, X, Y, ratio=0.5):

        self.instancesize, self.n_attrs = X.shape
        self.X = X
        self.Y = Y
        # self.ratio is the desired percentage of the rare instances,default 0.5
        self.ratio = ratio

    def over_sampling(self):

        # 获取原始数据集中normal instances和rare instances的个数
        normalinstancesize, rareinstanceX, rareinstanceY = self.refreshData(
            self.X, self.Y)
        rareinstancesize = self.instancesize - normalinstancesize

        # 如果需求的比例小于原始的percentage of the rare instances，意味着要删除原始的rareinstance，则不删除，直接返回原始的X，Y
        if self.ratio<rareinstancesize*1.0/(rareinstancesize+normalinstancesize):
            return X, Y

        else:
            p = int(((self.ratio -1) * rareinstancesize+ self.ratio* normalinstancesize)/(1-self.ratio) )
            # rareinstanceX, rareinstanceY 中抽取p个模块，加入到原始的数据集中。p也可能大于rareinstancesize
            keep=[]
            for i in range(p):
                keep.append(random.randint(0,rareinstancesize-1))

            traininginstancesX = rareinstanceX[keep]
            traininginstancesY = rareinstanceY[keep]
            '''
            print('number of normalinstances :', normalinstancesize)
            print('number of rareinstances :', rareinstancesize)
            print('p :', p)
            print('keep', keep)
            '''
            Y = np.hstack((self.Y, traininginstancesY))
            X = np.vstack((self.X, traininginstancesX))
            return X, Y

    def refreshData(self, dataX, dataY):
        '''
        dataX: 原始数据集的X
        dataY: 原始数据集的y
        返回缺陷数目为0的instances个数，返回缺陷数目大于0的instance的dataX 和 datay
        '''
        bugDataX = []
        bugDataY = []
        count = 0
        dataY = np.matrix(dataY).T
        dataX = np.array(dataX)
        for i in range(len(dataY)):
            if dataY[i] > 0:
                bugDataX.append(dataX[i])
                bugDataY.append(int(dataY[i]))
            else:
                count += 1

        bugDataX = np.array(bugDataX)
        bugDataY = np.array(bugDataY)
        return count, bugDataX, bugDataY


def main():
    X = np.array([[1, 1], [8, 8], [9, 9], [10, 10], [7, 9], [13, 13], [1, 2], [8, 2], [9, 2], [9, 2], [7, 2], [7, 2],
                  [3, 4], [4, 3], [6, 2], [7, 3], [3, 5], [4, 5], [6, 5], [7, 5], [2, 1], [1, 3], [1, 2], [4, 1], [1, 6], [3, 4], [4, 3], [6, 2], [7, 3], [3, 5], [4, 5], ])

    y = np.array([1, 2, 3, 0, 4, 5, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ])

    over_X, over_y = Oversampling(X=X, Y=y, ratio=0.4).over_sampling()


    print('over_X :', over_X)
    print('over_y :', over_y)


if __name__ == '__main__':
    main()