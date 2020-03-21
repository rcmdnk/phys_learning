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

    def run(self, cmd=None, verbose=0,
            signal='events/my_z.txt', bg='events/my_jj.txt'):
        """Main command"""
        PhysLearning(signal=signal, bg=bg, verbose=verbose).run(cmd)

    def multishot(self, shot=10, nvalue=3, max_val=16, verbose=0,
                  signal='events/my_z.txt', bg='events/my_jj.txt'):
        """Multishot"""
        PhysLearning(signal=signal, bg=bg, verbose=verbose).multishot(
            shot, nvalue, max_val)


def cli():
    """Main command line tool function."""
    if len(sys.argv) <= 1 or sys.argv[1] in ['-h', '--help']:
        PhysLearning.help()
    else:
        fire.Fire(CliObject)
