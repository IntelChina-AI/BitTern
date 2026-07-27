import os

import torch


CATQ_PARAMETERS_MANIFEST_FORMAT = "parameters_per_layer_v1"


class CatQParameterDiskStore:
    def __init__(self, manifest_path, manifest):
        self.manifest_path = os.path.realpath(manifest_path)
        self.base_dir = os.path.dirname(self.manifest_path)
        self.layers = {}
        for layer_id, path in manifest["layers"].items():
            if os.path.isabs(path):
                raise ValueError("Checkpoint manifest paths must be relative")
            resolved = os.path.realpath(os.path.join(self.base_dir, path))
            if os.path.commonpath((self.base_dir, resolved)) != self.base_dir:
                raise ValueError("Checkpoint manifest path escapes its checkpoint directory")
            self.layers[int(layer_id)] = resolved

    def __getitem__(self, layer_id):
        layer_path = self.layers[int(layer_id)]
        return torch.load(layer_path, map_location="cpu", weights_only=True)

    def __contains__(self, layer_id):
        return int(layer_id) in self.layers


def is_catq_parameter_manifest(data):
    return isinstance(data, dict) and data.get("__format__") == CATQ_PARAMETERS_MANIFEST_FORMAT


def load_catq_parameters(path):
    catq_parameters = torch.load(path, map_location="cpu", weights_only=True)
    if is_catq_parameter_manifest(catq_parameters):
        return CatQParameterDiskStore(path, catq_parameters)
    return catq_parameters
