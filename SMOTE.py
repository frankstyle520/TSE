# -*- coding: utf-8 -*-
# @Author: Kicc
# @Date:   2018-11-23 12:31:01
# @Last Modified by:   xiao yu
# @Last Modified time: 2019-3-25 14:42:05

from sklearn.neighbors import NearestNeighbors
import numpy as np


class Smote:
    def __init__(self, X, Y, ratio=0.5, k=5, r=2):
        """ ratio: [0.3-0.6],float
            k: [1-20], int
            r: [0.1, 5], float
        """
        self.instancesize, self.n_attrs = X.shape
        self.X = X
        self.Y = Y
        # self.ratio is the desired percentage of the rare instances, default 0.5
        self.ratio = ratio
        #k is the number of nearest neighbors of a defective module,default 5
        self.k = k
        #r is the power parameter for the minkowski distance metric，default 2
        self.r = r


    # 这个函数只返回合成的数据加上原始数据
    def over_sampling(self):

        normalinstancesize, rareinstanceX, rareinstanceY = self.refreshData(
            self.X, self.Y)
        rareinstancesize = self.instancesize - normalinstancesize

        # 如果需求的比例小于原始的percentage of the rare instances
        if self.ratio < rareinstancesize * 1.0 / (rareinstancesize + normalinstancesize):
            p = 0

        # 如果需要合成的缺陷模块个数p小于原始数据集中缺陷模块个数，则从原始数据集的缺陷模块中抽取p个，index=1
        elif self.ratio < 2.0 * rareinstancesize / (2.0 * rareinstancesize + normalinstancesize):
            p = int(((self.ratio - 1) * rareinstancesize + self.ratio * normalinstancesize) / (1 - self.ratio))
            keep = np.random.permutation(rareinstancesize)[:p]
            traininginstancesX = rareinstanceX[keep]
            traininginstancesY = rareinstanceY[keep]
            index = 1

        # 如果需要合成的缺陷模块个数p大于原始数据集中缺陷模块个数，则全部用上原始数据集的缺陷模块，并重复index次。
        else:
            p = rareinstancesize
            index = int(((self.ratio - 1) * rareinstancesize + self.ratio * normalinstancesize) / (
                        1 - self.ratio) / rareinstancesize)
            traininginstancesX = rareinstanceX
            traininginstancesY = rareinstanceY
        '''
        print('number of normalinstances :', normalinstancesize)
        print('number of rareinstances :', rareinstancesize)
        print('p :', p)
        print('index :', index)
        print('总共需要合成的rare instances为:', p * index)
        print('原始数据中rare instances为：', rareinstanceX, 'Y', rareinstanceY)
        '''

        # 总共需要合成的rare instances为p*index个
        # 如果需要合成的数据为0个，则直接返回原始的xy
        if (p == 0):
            return self.X, self.Y
        else:
            syntheticX=[]
            syntheticY=[]

            for i in range(p):
                # 使用自定义的取邻近k个函数，返回值是traininginstancesX[i]在rareinstanceX中的最近k个模块的下标nnarray
                nnarray = self.nearestNeighbors(
                    self.r, self.k, targetPoint=traininginstancesX[i], allPoints=rareinstanceX)
                '''
                print('选择的', p, '个traininginstancesX为', traininginstancesX, '\n,Y', traininginstancesY)
                print('\n选择第', i, '个traininginstanceX为',
                      traininginstancesX[i], ',它的近邻在rareinstance中的下角标为:', nnarray)
                for j in nnarray:
                    print('第', j, '个', rareinstanceX[j], rareinstanceY[j])
                '''
                syntheticiX, syntheticiY= self.populate(traininginstancesX[i], traininginstancesY[i], rareinstanceX, rareinstanceY, nnarray, self.r, index)
                syntheticX.append(syntheticiX)
                syntheticY.append(syntheticiY)
            syntheticX, syntheticY = np.asarray(syntheticX), np.asarray(syntheticY)

            #缺省值-1代表我不知道要给行（或者列）设置为几，reshape函数会根据原矩阵的形状自动调整。
            syntheticX = np.reshape(syntheticX, (-1, self.n_attrs))
            syntheticY = syntheticY.flatten()

            X = np.vstack((self.X, syntheticX))
            Y = np.hstack((self.Y, syntheticY))

            return X, Y

    # 从traininginstanceX的k个邻居中随机选取index次，生成index个合成的样本
    def populate(self, traininginstanceX, traininginstanceY, rareinstancesX, rareinstancesY, nnarray, r, index):
        syntheticX=[]
        syntheticY=[]
        for j in range(index):
            nn = np.random.randint(0, self.k)
            nn = min(nn, len(nnarray) - 1)
            dif = rareinstancesX[nnarray[nn]] - traininginstanceX
            gap = np.random.rand(1, self.n_attrs)
            syntheticinstanceX=traininginstanceX + gap.flatten() * dif
            syntheticX.append(syntheticinstanceX)

            '''
            print(
                '\n选择用于合成的', traininginstanceX, '一个近邻为的traininginstanceX为', rareinstancesX[nnarray[nn]])
            print(rareinstancesX[nnarray[nn]], '与',
                  traininginstanceX, '之间的dif :', dif)
            print('产生的随机向量为:', gap.flatten())
            print('合成的instance的向量为', traininginstanceX + gap.flatten() * dif)
            '''

            dist1 = (float)((np.sum(abs(syntheticinstanceX - traininginstanceX) ** r)) ** (1 / r))
            dist2 = (float)((np.sum(abs(syntheticinstanceX - rareinstancesX[nnarray[nn]]) ** r)) ** (1 / r))
            if (dist1 + dist2 != 0):
                syntheticinstanceY=(dist1 * rareinstancesY[nnarray[nn]] + dist2 * traininginstanceY) * 1.0 / (dist1 + dist2)
                syntheticY.append(syntheticinstanceY)
            else:
                syntheticinstanceY =traininginstanceY * 1.0
                syntheticY.append(syntheticinstanceY)
            '''
            print(traininginstanceX, traininginstanceY,
                  '和合成数据', syntheticinstanceX, '之间距离为', dist1)
            print(rareinstancesX[nnarray[nn]], rareinstancesY[nnarray[nn]],
                  '和合成数据', syntheticinstanceX, '之间距离为', dist2)
            print('合成的数据为：', syntheticinstanceX, syntheticinstanceY)
            '''
        return syntheticX, syntheticY


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

    def nearestNeighbors(self, r, k, targetPoint, allPoints):
        """获得距离目标点targetpoint最近的k个点的下标
           r: float
           k: int
           targetPoint: np.array[float]
           allPoints: List[np.array[float]]
            res = List[np.array[float]]
        """
        candidate = []
        index = 1 / r
        targetPoint = np.asarray(targetPoint)
        allPoints = np.asarray(allPoints)
        #如果rareinstances的数量小于k，则把一个点的所有最近邻都取上
        if k>len(allPoints):
            nearestneighbors=[i for i in range(len(allPoints))]
            #print('nearestneighbors:',nearestneighbors)
            return nearestneighbors

        else:
            for idx, point in enumerate(allPoints):
                subtraction = abs(point - targetPoint)
                result = np.sum(subtraction ** r)
                candidate.append((result ** index, idx))
            candidate = sorted(candidate, key=lambda x: x[0])
            res = [i[1] for i in candidate]

            return res[1:int(k + 1)]


def main():
    X = np.array([[1, 1], [8, 8], [9, 9], [10, 10], [7, 9], [13, 13], [1, 2], [8, 2], [9, 2], [9, 2], [7, 2], [7, 2],
                  [3, 4], [4, 3], [6, 2], [7, 3], [3, 5], [4, 5], [6, 5], [7, 5], [2, 1], [1, 3], [1, 2], [4, 1],
                  [1, 6], [3, 4], [4, 3], [6, 2], [7, 3], [3, 5], [4, 5], ])

    y = np.array([2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ])

    smote_X, smote_y = Smote(X=X, Y=y, ratio=0.5, k=5, r=1).over_sampling()

    print('smote_X :', smote_X)
    print('smote_y :', smote_y)


if __name__ == '__main__':
    main()
