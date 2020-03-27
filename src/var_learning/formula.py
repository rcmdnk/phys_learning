import numpy as np
from numbers import Number


class Formula():
    def __init__(self, n_values, min_use=1, max_use=1, var_labels=None,
                 seed=None, fix_dim=False):
        self.n_values = n_values
        self.min_use = min_use
        self.max_use = max_use
        if var_labels is not None:
            self.var_labels = var_labels
        else:
            self.var_labels = []
            for i in range(n_values):
                self.var_labels.append('x{}'.format(i))

        self.seed = seed
        self.rand = np.random.RandomState(self.seed)

        self.fix_dim = fix_dim
        self.rpn = []

        self.operators = [lambda x, y: x + y, lambda x, y: x - y,
                          lambda x, y: x * y, lambda x, y: self.div(x, y)]
        self.symbols = ["+", "-", "*", "/"]
        self.n_operators = len(self.operators)

    def div(self, x, y):
        if isinstance(x, Number):
            return 0 if y == 0 else x / y
        return np.divide(x, y, out=np.zeros_like(x), where=y!=0)

    def set_seed(self, seed=0):
        self.seed = seed
        self.rand.seed(self.seed)

    def make_rpn(self):
        self.rpn = []
        stack = []
        n = self.rand.randint(self.min_use, self.max_use + 1)
        for i in range(n):
            val = self.rand.randint(0, self.n_values)
            self.rpn.append(val)
            stack.append(1)
            while len(stack) > 1:
                ope = self.rand.randint(0, self.n_operators + 1)
                if ope == self.n_operators:
                    break
                if self.symbols[ope] in ("+", "-"):
                    if self.fix_dim and stack[-1] != stack[-2]:
                        break
                    stack.pop()
                elif self.symbols[ope] == "*":
                    last_dim = stack.pop()
                    stack[-1] += last_dim
                else:
                    last_dim = stack.pop()
                    stack[-1] -= last_dim
                self.rpn.append(ope - self.n_operators)
        while len(stack) > 1:
            ope = self.rand.randint(0, self.n_operators)
            if ope == self.n_operators:
                break
            self.rpn.append(ope - self.n_operators)
            stack.pop()

    def get_formula(self):
        stack = []
        for i in self.rpn:
            if i >= 0:
                stack.append(self.var_labels[i])
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
