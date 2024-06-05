import sys
import lr_gym
import importlib

def launcher():
    command = sys.argv[1]
    mod = importlib.import_module("lr_gym."+command)
    sys.argv = [str(mod.__file__)]+sys.argv[2:]
    mod.main()