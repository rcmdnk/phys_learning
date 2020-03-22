"""
    Functions for two particles
"""
import numpy as np
from .phys import pt


class TwoParticles():

    def __init__(self, data, is_signal=1):
        self.var_labels = ["px1", "py1", "pz1", "e1",
                           "px2", "py2", "pz2", "e2"]
        self.data = self.get_data(data, is_signal)

    def get_data(self, data, is_signal):
        if type(data) == str:
            with open(data) as f:
                lines = f.readlines()
        else:
            lines = data
        output = []
        for line in lines:
            if type(line) == str:
                line = line.strip()
                line = [float(x) for x in line.split()]
            if len(line) == 9:
                is_signal = line[8]
            if pt(line[0], line[1]) > pt(line[4], line[5]):
                output.append(line + [is_signal])
            else:
                output.append(line[4:8] + line[0:4] + [is_signal])
        return np.array(output)
