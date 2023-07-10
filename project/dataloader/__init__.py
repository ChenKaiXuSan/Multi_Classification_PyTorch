import sys 

try:
    from data_loader import *
except:
    sys.path.append('/workspace/Multi_Classification_PyTorch/project/dataloader')
    from data_loader import *