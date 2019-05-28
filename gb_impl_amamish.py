#coding=utf-8

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np


# Ваш email, который вы укажете в форме для сдачи
AUTHOR_EMAIL = 'amamish@yandex.ru'
# Параметрами с которыми вы хотите обучать деревья
TREE_PARAMS_DICT = {'max_depth': 7}
# Параметр tau (learning_rate) для вашего GB
TAU = 0.05

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def der_log_loss(y_hat, y_true):
    return (y_true - y_hat) / ((1 - y_hat) * y_hat)

def calc_gradient(y_hat, y_true):
    return der_log_loss(sigmoid(y_hat), y_true) * der_sigmoid(y_hat)

class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
        
    def fit(self, X_data, y_data):
        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, y_data)
        self.estimators = []
        curr_pred = self.base_algo.predict(X_data)
        for iter_num in range(self.iters):
            # Нужно посчитать градиент функции потерь
            grad = calc_gradient(curr_pred, y_data) # TODO
            # Нужно обучить DecisionTreeRegressor предсказывать антиградиент
            # Не забудьте про self.tree_params_dict
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, grad) # TODO
            self.estimators.append(algo)
            # Обновите предсказания в каждой точке
            curr_pred += self.tau * algo.predict(X_data) # TODO
        return self
    
    def predict(self, X_data):
        # Предсказание на данных
        res = self.base_algo.predict(X_data)
        for estimator in self.estimators:
            res += self.tau * estimator.predict(X_data)
        # Задача классификации, поэтому надо отдавать 0 и 1
        return res > 0.
