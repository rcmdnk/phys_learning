import numpy as np
from .plot import histTwo


def mass(p1, p2):
    if type(p1) == np.ndarray and p1.ndim >= 2:
        x = p1[:, 0] + p2[:, 0]
        y = p1[:, 1] + p2[:, 1]
        z = p1[:, 2] + p2[:, 2]
        e = p1[:, 3] + p2[:, 3]
    else:
        x = p1[0] + p2[0]
        y = p1[1] + p2[1]
        z = p1[2] + p2[2]
        e = p1[3] + p2[3]
    return np.sqrt(e**2 - x**2 - y**2 - z**2)


def pt(x, y):
    return np.sqrt(x**2 + y**2)


def histTwoPhys(data1, data2, name="phys"):
    histTwo(mass(data1[:, 0:4], data1[:, 4:8]),
            mass(data2[:, 0:4], data2[:, 4:8]),
            80, [0, 200], name=name + '_mjj', xlabel='$M_{jj}$ (GeV)',
            label1='signal', label2='bg')
    histTwo(data1[:, 0], data2[:, 0], 100, [0, 500],
            name + '_xj1', '$x_{j1}$ (GeV)',
            label1='signal', label2='bg')
    histTwo(data1[:, 4], data2[:, 4], 100, [0, 500],
            name + '_xj2', '$x_{j2}$ (GeV)',
            label1='signal', label2='bg')
    histTwo(data1[:, 3], data2[:, 3], 100, [0, 500],
            name + '_Ej1', '$E_{j1}$ (GeV)',
            label1='signal', label2='bg')
    histTwo(data1[:, 7], data2[:, 7], 100, [0, 500],
            name + '_Ej2', '$E_{j2}$ (GeV)',
            label1='signal', label2='bg')

    histTwo(pt(data1[:, 0], data1[:, 1]), pt(data2[:, 0], data2[:, 1]),
            100, [0, 500], name + '_ptj1', '$P_{Tj1}$ (GeV)',
            label1='signal', label2='bg')
    histTwo(pt(data1[:, 4], data1[:, 5]), pt(data2[:, 4], data2[:, 5]),
            100, [0, 500], name + '_ptj2', '$P_{Tj2}$ (GeV)',
            label1='signal', label2='bg')
