"""Command-line entry points; heavy model dependencies are loaded only on demand."""
import sys
def main():
    commands = ['replay', 'predict', 'train', 'timing', 'verify']
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print('Usage: python -m amp_classification {' + ','.join(commands) + '} [options]')
        raise SystemExit(0 if len(sys.argv) == 1 or sys.argv[1] in ['-h', '--help'] else 2)
    import importlib
    module = importlib.import_module('amp_classification.' + sys.argv[1])
    module.main(sys.argv[2:])
if __name__ == '__main__': main()
