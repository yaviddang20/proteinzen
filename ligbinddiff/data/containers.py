""" Generic data storage objects to allow for interfacing between different types of modules """
import torch


from ligbinddiff.utils.treeops import treemap


class Protein:

    NODE_DATA = [
        'x',
        'x_mask',
        'x_cb',
    ]

    EDGE_DATA = [
        'distance'
    ]

    def __init__(self, data):
        self.data = data

    def __getattribute__(self, name):
        if name in self.data.keys():
            return self.data[name]
        else:
            return self.__getattr__(name)

    def to(self, device):
        self.data = treemap(lambda x: x.to(device), self.data)
