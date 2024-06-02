import torch
import safetensors
import torch
import random
from torchvision import transforms as T
import torchvision
# from sklearn.cluster import KMeans


def open_weights(path):
    try: # if ".safetensors" in path:
        state_dict = {}
        with safetensors.safe_open(path, "pt") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    except:
        state_dict = torch.load(path, map_location="cpu")

    return state_dict



