"""
    Core module to provides gcpm functions.
"""

import sys
from .__init__ import __version__
from .__init__ import __program__


class PhysLearning():
    """Phys Learning"""

    def __init__(self):
        pass

    @staticmethod
    def help():
        print("""
Usage: phys_learning [--config=<config>] [--test=<test>] <command>

    commands:
        run        : Run user process.
        version    : Show version.
        help       : Show this help.
""")

    @staticmethod
    def version():
        print("%s: %s" % (__program__, __version__))

    def run(self):
        print(self.__class__.__name__, sys._getframe().f_code.co_name)
