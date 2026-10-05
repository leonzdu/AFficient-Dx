import argparse
import importlib
import sys

def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    modules = {'train': 'training', 'study': 'study', 'compare': 'benchmarks', 'tables': 'paper_tables', 'verify': 'verification', 'predict': 'prediction'}
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=modules)
    if not args or args[0] in ('-h', '--help'):
        parser.parse_args(args)
        return
    selected = parser.parse_args(args[:1]).command
    importlib.import_module('afficient_dx.' + modules[selected]).main(args[1:])
