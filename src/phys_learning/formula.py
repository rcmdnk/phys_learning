import numpy as np


class Formula():
    def __init__(self, n_values, min_use=1, max_use=1, var_labels=None,
                 seed=None):
        self.n_values = n_values
        self.min_use = min_use
        self.max_use = max_use
        if var_labels is not None:
            self.var_labels = var_labels
        else:
            self.var_lables = []
            for i in n_values:
                self.var_labels.append('x{}'.format(i))

        self.seed = seed
        self.rand = np.random.RandomState(self.seed)
        self.rpn = []

        self.operators = [lambda x, y: x + y, lambda x, y: x - y,
                          lambda x, y: x * y, lambda x, y: x / y]
        self.symbols = ["+", "-", "*", "/"]
        self.n_operators = len(self.operators)

    def set_seed(self, seed=0):
        self.seed = seed
        self.rand.seed(self.seed)

    def make_rpn(self):
        self.rpn = []
        stack = 0
        n = self.rand.randint(self.min_use, self.max_use + 1)
        for i in range(n):
            val = self.rand.randint(0, self.n_values)
            self.rpn.append(val)
            stack += 1
            while stack > 1:
                ope = self.rand.randint(0, self.n_operators + 1)
                if ope == self.n_operators:
                    break
                self.rpn.append(ope - self.n_operators)
                stack -= 1
        while stack > 1:
            ope = self.rand.randint(0, self.n_operators)
            if ope == self.n_operators:
                break
            self.rpn.append(ope - self.n_operators)
            stack -= 1

    def formula(self):
        stack = []
        for i in self.rpn:
            if i >= 0:
                stack.append(self.var_lables[i])
            else:
                v2 = stack.pop()
                v1 = stack.pop()
                stack.append("({} {} {})".format(v1, self.symbols[i], v2))
        return stack[0]

    def calc(self, data):
        data = np.array(data, dtype='float')
        if data.ndim == 1:
            data = np.array([data])

        stack = []
        for i in self.rpn:
            if i >= 0:
                stack.append(data[:, i:i + 1])
            else:
                val = stack.pop()
                stack[-1] = self.operators[i](stack[-1], val)
        return stack[0]
