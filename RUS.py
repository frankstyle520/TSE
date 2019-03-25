from sklearn.neighbors import NearestNeighbors
import numpy as np
import random


class RUS:
    def __init__(self, X, Y, ratio=0.5):
        self.X = X
        self.Y = Y
        # self.ratio is the desired percentage of the rare instances, default 0.5
        self.ratio = ratio

    # 这个函数只返回合成的数据
    def under_sampling(self):

        Defectivecount, DefectiveX, DefectiveY, Non_Defectivecount, Non_DefectiveX, Non_DefectiveY = self.refreshData(
            self.X, self.Y)
        if self.ratio < Defectivecount / Non_Defectivecount:
            p = 0
        else:
            p = int(Non_Defectivecount - Defectivecount / self.ratio)

        # 随机的从Non_DefectiveX, Non_DefectiveY删除p个实例。返回删除了p个实例的Non_DefectiveX, Non_DefectiveY，加上DefectiveX, DefectiveY
        np.random.seed(Defectivecount)
        np.random.shuffle(Non_DefectiveX)
        np.random.seed(Defectivecount)
        np.random.shuffle(Non_DefectiveY)
        remainNonX, remainNony = Non_DefectiveX[p:], Non_DefectiveY[p:]
        rusX, rusY = np.append(remainNonX, DefectiveX, axis=0), np.append(
            remainNony, DefectiveY, axis=0)
        return rusX, rusY

    def refreshData(self, dataX, dataY):
        """ The number of defection which greater  than zero: DetectiveX, y
            The number equals zero: Non_DefectiveX, y
            Both count of the above two.

        """
        DefectiveX = []
        DefectiveY = []
        Non_DefectiveX = []
        Non_DefectiveY = []
        Defectivecount = 0
        Non_Defectivecount = 0
        dataY = np.matrix(dataY).T
        dataX = np.array(dataX)
        for i in range(len(dataY)):
            if dataY[i] > 0:
                Defectivecount += 1
                DefectiveX.append(dataX[i])
                DefectiveY.append(int(dataY[i]))
            else:
                Non_Defectivecount += 1
                Non_DefectiveX.append(dataX[i])
                Non_DefectiveY.append(int(dataY[i]))

        DefectiveX = np.array(DefectiveX)
        DefectiveY = np.array(DefectiveY)
        Non_DefectiveX = np.array(Non_DefectiveX)
        Non_DefectiveY = np.array(Non_DefectiveY)
        return Defectivecount, DefectiveX, DefectiveY, Non_Defectivecount, Non_DefectiveX, Non_DefectiveY


def main():
    X = np.array([[1, 1], [8, 8], [9, 9], [10, 10], [7, 9], [13, 13], [1, 2], [8, 2], [9, 2], [9, 2], [7, 2], [7, 2],
                  [3, 4], [4, 3], [6, 2], [7, 3], [3, 5], [4, 5], [6, 5], [7, 5], [2, 1], [1, 3], [1, 2], [4, 1], [1, 6], [3, 4], [4, 3], [6, 2], [7, 3], [3, 5], [4, 5], ])

    y = np.array([2, 3, 5, 1, 3, 1, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ])

    rus_X, rus_y = RUS(X=X, Y=y, ratio=0.1).under_sampling()

    print('rus_X :', rus_X)
    print('rus_y :', rus_y)


if __name__ == '__main__':
    main()
