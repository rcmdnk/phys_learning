"""
    Command line interface for core object
"""


import sys
import fire
from .__init__ import __version__, __program__
from .phys_binary import PhysBinary


class CliObject(object):
    """CliObject"""

    def __init__(self):
        pass

    def version(self):
        """Show version."""
        print("%s: %s" % (__program__, __version__))

    def run(self, cmd=None, verbose=0, shot=10, nvalue=3,
            method='DNN',
            signal='events/my_z.txt', bg='events/my_jj.txt'):
        """Main command"""
        PhysBinary(signal=signal, bg=bg, shot=shot, nvalue=nvalue,
                   method=method, verbose=verbose,).run(cmd)


def cli():
    """Main command line tool function."""
    if len(sys.argv) <= 1 or sys.argv[1] in ['-h', '--help', 'help']:
        print("%s: %s" % (__program__, 'Usage: phys_learning run <cmd>'))
    else:
        fire.Fire(CliObject)
