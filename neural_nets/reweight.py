
import models as models
import data as data
import save as s
from sklearn.metrics import classification_report
from keras.models import load_model
import argparse

from train import evaluate
from discretize import scale_round, custom_scale




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='To be announced...')

    parser.add_argument('--h5model')
    parser.add_argument("--batchsize",default=32,type=int)
    parser.add_argument("--scale",default=100,type=int)
    parser.add_argument("--tolerance",default=3,type=int)
    parser.add_argument("--savedir",default="",type=str)
    parser.add_argument("--save",action="store_true")
    parser.add_argument("--custom",action="store_true")


    args = parser.parse_args()

    batch_size = args.batchsize
    savedir = args.savedir
    scale = args.scale
    tolerance = args.tolerance

    h5_file_name = args.h5model
    model = load_model(h5_file_name)

    if "MNIST" in args.h5model:
        flatten_inputs = "MLP" in args.h5model
        dataset = data.load_mnist(flatten_inputs)
    else:
        dataset = data.load_cifar()

    x_train = dataset["x_train"]
    x_test = dataset["x_test"]

    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    print("\nINFO: Evaluating original model")
    accuracy = evaluate(model,x_test,y_test,batch_size)

    weights = model.get_weights()
    print(weights[-1])

    if args.custom:
        discrete_weights = custom_scale(weights,tolerance)
    else:
        discrete_weights = scale_round(weights,scale)

    model.set_weights(discrete_weights)

    print(discrete_weights[-1])
    print(model.get_weights()[-1])

    print("\nINFO: Evaluating discretized model")
    evaluate(model,x_test,y_test,batch_size)

    if args.save:
        print("\nINFO: Saving trained model files to {}".format(savedir))
        s.save_model(model,savedir)
        s.save_weights(discrete_weights,savedir)

