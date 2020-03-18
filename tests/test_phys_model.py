import pytest
import numpy as np
from phys_learning.phys_model import PhysModel


@pytest.mark.phys_model
def test_version():
    pm = PhysModel(16, 5)
    pm.make_rpn()
    rpn = pm.get_rpn()
    formula = pm.get_formula()
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
    result = pm.calc(data)
    print(result)

    assert rpn == [3, 6, -4, 1, -4, 4, -1, 0, -4, 7, 7, -3, -4, 3, 5, -1, -3,
                   7, 6, -2, -3, 1, -2, 1, -3, 6, -3, 2, -4, 6, -2]
    assert formula == '((((((((((((e1 + pz2) + py1) / px2) + px1)' \
        ' + (e2 - e2)) - (e1 / py2)) - (e2 * pz2)) * py1) - py1) - pz2)' \
        ' + pz1) * pz2)'
    assert result[0][0] == -784.9333333333333
    assert result[1][0] == 103.66666666666667
