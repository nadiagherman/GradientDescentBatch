import random


class MyGDregressor:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    # Batch Gradient Descent
    def fit(self, x, y, learningRate=0.001, noEpochs=10000):
        self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]
        # self.coef_ = [random.random() for _ in range(len(x[1]))]

        for epoch in range(noEpochs):
            grdntsums = [0.0 for _ in range(len(x[0]) + 1)]

            for j in range(len(x[0])):
                gradient = 0.0
                for i in range(len(x)):
                    # crtError = sum([self.eval(x[i]) - y[i] for i in range(len(x))])
                    # crtError = 1.0 / len(x) * crtError

                    crtError = self.eval(x[i]) - y[i]
                    gradient += crtError * x[i][j]

                grdntsums[j] = 1.0 / len(x) * gradient
            grdntsums[len(x[0])] = 1.0 / len(x) * sum([self.eval(x[i]) - y[i] for i in range(len(x))])

            for j in range(0, len(x[0])):
                self.coef_[j] = self.coef_[j] - learningRate * grdntsums[j]
            self.coef_[len(x[0])] = self.coef_[len(x[0])] - learningRate * grdntsums[len(x[0])]

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def eval(self, datax):
        valoutput = self.coef_[-1]
        for j in range(len(datax)):
            valoutput += self.coef_[j] * datax[j]
        return valoutput

    def evalafter(self, datax):
        valoutput = self.intercept_
        for j in range(len(datax)):
            valoutput += self.coef_[j] * datax[j]
        return valoutput

    def predict(self, x):
        yComputed = [self.evalafter(xi) for xi in x]
        return yComputed
