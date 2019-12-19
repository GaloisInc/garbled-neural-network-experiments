import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        return json.JSONEncoder.default(self, obj)

savedir = "/Users/nls/repos/fancy-garbling/garbled-neural-net-experiments/neural_nets/cnntanh/"


def save_model(model,savedir):
    with open(savedir + "model.json","w") as f:
        f.write(model.to_json())

def save_weights(weights,savedir):
    with open(savedir + "weights.json","w") as f:
        json.dump(weights,f,cls=NumpyEncoder)

def save_x_test(x_test,savedir):
    with open(savedir + "tests.json","w") as f:
        json.dump(x_test.astype(int),f,cls=NumpyEncoder)

def save_y_test(y_test,savedir):
    with open(savedir + "labels.json","w") as f:
        json.dump(y_test.astype(int),f,cls=NumpyEncoder)



