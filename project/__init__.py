import sys

try:
    from main import *
except:
    sys.path.append('/workspace/Multi_Classification_PyTorch/project')
    from main import *