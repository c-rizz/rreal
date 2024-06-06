import sys
import adarl
import importlib

def launcher():
    command = sys.argv[1]
    mod = importlib.import_module("adarl."+command)
    sys.argv = [str(mod.__file__)]+sys.argv[2:]
    mod.main()