import random


class MyGDregressorBi:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    # Batch Gradient Descent
    def fit(self, x, y, learningRate=0.001, noEpochs=10000):
        self.coef_ = [0.0, 0.0, 0.0]

        for epoch in range(noEpochs):

            grdntsums = [0.0, 0.0, 0.0]

            for j in range(2):
                gradient = 0.0
                for i in range(len(x)):
                    crtError = (self.eval(x[i]) - y[i])
                    gradient += crtError * x[i][j]

                grdntsums[j] = 1.0 / len(x) * gradient
            grdntsums[2] = 1.0 / len(x) * sum([(self.eval(x[i]) - y[i]) for i in range(len(x))])

            for j in range(0, 2):
                self.coef_[j] = self.coef_[j] - learningRate * grdntsums[j]
            self.coef_[2] = self.coef_[2] - learningRate * grdntsums[2]

        self.intercept_ = self.coef_[2]
        self.coef_ = self.coef_[:2]

    def eval(self, datax):

        return self.coef_[2] + self.coef_[0] * datax[0] + self.coef_[1] * datax[1]

    def evalafter(self,datax):

        return self.intercept_ + self.coef_[0] * datax[0] + self.coef_[1] * datax[1]

    def predict(self, x):

        yComputed = [self.evalafter(xi) for xi in x]
        return yComputed

