"""
    Physics Model
"""

import numpy as np


class Operators():
    @staticmethod
    def add(x, y):
        return x + y

    @staticmethod
    def sub(x, y):
        return x - y

    @staticmethod
    def mul(x, y):
        return x * y

    @staticmethod
    def div(x, y):
        return x / y


class PhysModel():
    """Phys Model"""

    OPERATORS = [Operators.add, Operators.sub, Operators.mul, Operators.div]
    VALUES = ["px1", "py1", "pz1", "e1", "px2", "py2", "pz2", "e2"]
    SYMBOLS = ["+", "-", "*", "/"]

    def __init__(self, n=1, seed=None):
        self.n = n
        self.rand = np.random.RandomState(seed)
        self.rpn = []

        self.n_operators = len(PhysModel.OPERATORS)
        self.n_values = len(PhysModel.VALUES)

    def set_seed(self, seed=0):
        self.rand.seed(seed)

    def make_rpn(self):
        self.rpn = []
        stack = 0
        for i in range(self.n):
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

    def get_rpn(self):
        return self.rpn

    def get_formula(self):
        stack = []
        for i in self.rpn:
            if i >= 0:
                stack.append(PhysModel.VALUES[i])
            else:
                v2 = stack.pop()
                v1 = stack.pop()
                stack.append("({} {} {})".format(v1, PhysModel.SYMBOLS[i], v2))
        return stack[0]

    def calc(self, data):
        data = np.array(data, dtype='float')

        stack = []
        for i in self.rpn:
            if i >= 0:
                stack.append(data[:, i:i + 1])
            else:
                val = stack.pop()
                stack[-1] = PhysModel.OPERATORS[i](stack[-1], val)
        return stack[0]
