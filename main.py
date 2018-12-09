# -*- coding: utf-8 -*-
# @Author: Kicc
# @Date:   2018-12-06 13:00:17
# @Last Modified by:   Kicc
# @Last Modified time: 2018-12-06 13:18:59

import numpy as np
from SMOTEDE import SMOTEDE
from RUSDE import RUSDE
from RUS import RUS
from Processing import Processing
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from Smote import Smote
from PerformanceMeasure import PerformanceMeasure

def bootstrap(dataset):

    training_data_X, training_data_y, testing_data_X, testing_data_y = Processing(
    ).separate_data(dataset)


    brrde = SMOTEDE(NP=10, F=0.6, CR=0.7, generation=50,len_x=3,ratiovalue_up_range=1.5,ratiovalue_down_range=0.5,kRange=list(range(1, 21)),
            rvalue_up_range=5.0,rvalue_down_range=0.1, X=training_data_X, y=training_data_y, classifer=BayesianRidge())
    brrmaxpara=brrde.process()
    smote_X, smote_y = Smote(training_data_X, training_data_y, ratio=brrmaxpara[0], k=brrmaxpara[1], r=brrmaxpara[2]).over_sampling()
    brr = BayesianRidge().fit(smote_X, smote_y)
    brr_pred_y = brr.predict(testing_data_X)
    brr_fpa = PerformanceMeasure(testing_data_y, brr_pred_y).FPA()
    print('smoteDE_brr_fpa:', brr_fpa)


    rfde = SMOTEDE(NP=10, F=0.6, CR=0.7, generation=50,len_x=3,ratiovalue_up_range=1.5,ratiovalue_down_range=0.5,kRange=list(range(1, 21)),
            rvalue_up_range=5.0,rvalue_down_range=0.1, X=training_data_X, y=training_data_y, classifer=RandomForestRegressor())
    rfmaxpara=rfde.process()
    smote_X, smote_y = Smote(training_data_X, training_data_y, ratio=rfmaxpara[0], k=rfmaxpara[1], r=rfmaxpara[2]).over_sampling()
    rf=RandomForestRegressor().fit(smote_X, smote_y)
    rf_pred_y = rf.predict(testing_data_X)
    rf_fpa = PerformanceMeasure(testing_data_y, rf_pred_y).FPA()
    print('smoteDE_rf_fpa:', rf_fpa)


    brrde = RUSDE(NP=10, F=0.6, CR=0.7, generation=50,len_x=1,ratiovalue_up_range=1.5,ratiovalue_down_range=0.5,
             X=training_data_X, y=training_data_y, classifer=BayesianRidge())
    brrmaxpara=brrde.process()
    rus_X, rus_y = RUS(training_data_X, training_data_y, ratio=brrmaxpara[0]).under_sampling()
    brr = BayesianRidge().fit(rus_X, rus_y)
    brr_pred_y = brr.predict(testing_data_X)
    brr_fpa = PerformanceMeasure(testing_data_y, brr_pred_y).FPA()
    print('rusDE_brr_fpa:', brr_fpa)

    rfde = RUSDE(NP=10, F=0.6, CR=0.7, generation=50,len_x=1,ratiovalue_up_range=1.5,ratiovalue_down_range=0.5,
             X=training_data_X, y=training_data_y, classifer=RandomForestRegressor())
    rfmaxpara=rfde.process()
    rus_X, rus_y = RUS(training_data_X, training_data_y, ratio=rfmaxpara[0]).under_sampling()
    rf = RandomForestRegressor().fit(rus_X, rus_y)
    rf_pred_y = rf.predict(testing_data_X)
    rf_fpa = PerformanceMeasure(testing_data_y, rf_pred_y).FPA()
    print('rusDE_rf_fpa:', rf_fpa)

    smote_X, smote_y = Smote(training_data_X, training_data_y, ratio=1.0, k=5, r=2).over_sampling()
    brr = BayesianRidge().fit(smote_X, smote_y)
    brr_pred_y = brr.predict(testing_data_X)
    brr_fpa = PerformanceMeasure(testing_data_y, brr_pred_y).FPA()
    print('smote_brr_fpa:', brr_fpa)

    rf = RandomForestRegressor().fit(smote_X, smote_y)
    rf_pred_y = rf.predict(testing_data_X)
    rf_fpa = PerformanceMeasure(testing_data_y, rf_pred_y).FPA()
    print('smote_rf_fpa:', rf_fpa)


    rus_X, rus_y = RUS(training_data_X, training_data_y, ratio=1.0).under_sampling()
    brr = BayesianRidge().fit(rus_X, rus_y)
    brr_pred_y = brr.predict(testing_data_X)
    brr_fpa = PerformanceMeasure(testing_data_y, brr_pred_y).FPA()
    print('rus_brr_fpa:', brr_fpa)

    rf = RandomForestRegressor().fit(rus_X, rus_y)
    rf_pred_y = rf.predict(testing_data_X)
    rf_fpa = PerformanceMeasure(testing_data_y, rf_pred_y).FPA()
    print('rus_rf_fpa:', rf_fpa)

    brr = BayesianRidge().fit(training_data_X, training_data_y)
    brr_pred_y = brr.predict(testing_data_X)
    brr_fpa = PerformanceMeasure(testing_data_y, brr_pred_y).FPA()
    print('brr_fpa:', brr_fpa)

    rf=RandomForestRegressor().fit(training_data_X, training_data_y)
    rf_pred_y = rf.predict(testing_data_X)
    rf_fpa = PerformanceMeasure(testing_data_y, rf_pred_y).FPA()
    print('rf_fpa:', rf_fpa)


if __name__ == '__main__':
    for dataset, filename in Processing().import_single_data():
        print (filename)
        bootstrap(dataset=dataset)

