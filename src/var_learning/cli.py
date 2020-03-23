"""
    Command line interface for core object
"""


import sys
import fire
from .__init__ import __version__, __program__
from .core import VarLearning


class CliObject(object):
    """CliObject"""

    def __init__(self):
        pass

    def version(self):
        """Show version."""
        print("%s: %s" % (__program__, __version__))

    def run(self, cmd=None, data='events/data.txt', **kw):
        """Main command"""
        VarLearning(data=data, **kw).run(cmd)


def cli():
    """Main command line tool function."""
    if len(sys.argv) <= 1 or sys.argv[1] in ['-h', '--help', 'help']:
        print("%s: %s" % (__program__, 'Usage: var_learning run <cmd>'))
    else:
        fire.Fire(CliObject)
