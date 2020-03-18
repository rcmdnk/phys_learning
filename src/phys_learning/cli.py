"""
    Command line interface for core object
"""


import sys
from .core import PhysLearning
import fire


class CliObject(object):
    """CliObject"""

    def __init__(self):
        pass

    def help(self):
        """Show help."""
        PhysLearning.help()

    def version(self):
        """Show version."""
        PhysLearning.version()

    def run(self):
        """Main command"""
        print(self.__class__.__name__, sys._getframe().f_code.co_name)
        PhysLearning('events/my_z.txt', 'events/my_jj.txt').run()


def cli():
    """Main command line tool function."""
    if len(sys.argv) <= 1 or sys.argv[1] in ['-h', '--help']:
        PhysLearning.help()
    else:
        fire.Fire(CliObject)
