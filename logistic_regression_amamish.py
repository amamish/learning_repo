# coding=utf-8
import numpy as np
from sklearn.base import BaseEstimator


# Ваш email, который вы укажете в форме для сдачи
AUTHOR_EMAIL = 'amamish@yandex.ru'

LR_PARAMS_DICT = {
    'C': 10.,
    'random_state': 777,
    'iters': 1000,
    'batch_size': 2000,
    'step': 0.01
}

class MyLogisticRegression(BaseEstimator):
    def __init__(self, C, random_state, iters, batch_size, step):
        self.C = C
        self.random_state = random_state
        self.iters = iters
        self.batch_size = batch_size
        self.step = step

    # будем пользоваться этой функцией для подсчёта <w, x>
    def __predict(self, X):
        return np.dot(X, self.w) + self.w0

    # sklearn нужно, чтобы predict возвращал классы, поэтому оборачиваем наш __predict в это
    def predict(self, X):
        res = self.__predict(X)
        res[res > 0] = 1
        res[res < 0] = 0
        return res

    # производная регуляризатора
    def der_reg(self):
        res = - 1 / self.C * (np.sign(self.w) + self.w)
        # TODO
        return res

    # будем считать стохастический градиент не на одном элементе, а сразу на пачке (чтобы было эффективнее)
    def der_loss(self, x, y):
        # x.shape == (batch_size, features)
        # y.shape == (batch_size,)

        # считаем производную по каждой координате на каждом объекте
        # TODO
        y_hat = self.predict(x)
        ders_w = - (1 / self.C) * np.dot(x.T,(y - y_hat))
        #der_log_loss(sigmoid(y_hat), y) * der_sigmoid(y_hat)
        der_w0 = (1 / self.C) *(y_hat - y)

        # для масштаба возвращаем средний градиент по пачке
        # TODO
        return ders_w, np.mean(der_w0)

    def fit(self, X_train, y_train):
        # RandomState для воспроизводитмости
        random_gen = np.random.RandomState(self.random_state)
        
        # получаем размерности матрицы
        size, dim = X_train.shape
        
        # случайная начальная инициализация
        self.w = random_gen.rand(dim)
        self.w0 = random_gen.randn()

        for _ in range(self.iters):  
            # берём случайный набор элементов
            rand_indices = random_gen.choice(size, self.batch_size)
            # исходные метки классов это 0/1
            x = X_train[rand_indices]
            y = y_train[rand_indices]

            # считаем производные
            der_w, der_w0 = self.der_loss(x, y)
            der_w += self.der_reg()

            # обновляемся по антиградиенту
            self.w -= der_w * self.step
            self.w0 -= der_w0 * self.step

        # метод fit для sklearn должен возвращать self
        return self