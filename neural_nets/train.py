
import models as models
import data as data
import save as s
import time
from sklearn.metrics import classification_report
import argparse

from discretize import scale_round


def evaluate(model,x_test,y_test,batch_size):
    try:
        print()
        print(classification_report(y_test.argmax(axis=1),predictions.argmax(axis=1)))
    except:
        print("WARNING: unable to compute classification report")

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score[1]


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='To be announced...')

    parser.add_argument('--model',
                        choices=["SecureML_MLP_MNIST",
                                 "CryptoNets_CNN_MNIST",
                                 "MiniONN_CNN_MNIST",
                                 "MiniONN_CNN_CIFAR",
                                 "DeepSecure_CNN_MNIST",
                                 "test_MLP_MNIST"])
    parser.add_argument("--batchsize",default=32,type=int)
    parser.add_argument("--epochs",default=100,type=int)
    parser.add_argument("--savedir",type=str)
    parser.add_argument("--alternate",action="store_true")

    args = parser.parse_args()
    print("alternate set to: ",args.alternate)
    batch_size = args.batchsize
    epochs = args.epochs
    savedir = args.savedir

    if "MNIST" in args.model:
        flatten_inputs = "MLP" in args.model
        dataset = data.load_mnist(flatten_inputs)
    else:
        dataset = data.load_cifar()

    if args.model == "SecureML_MLP_MNIST":
        model = models.SecureML_MLP_MNIST(alternate=args.alternate)
        discrete_model = models.SecureML_MLP_MNIST()

        scale = 10

    elif args.model == "CryptoNets_CNN_MNIST":
        model = models.CryptoNets_CNN_MNIST(pooling="max",alternate=args.alternate)
        discrete_model = models.CryptoNets_CNN_MNIST(pooling="max",alternate=args.alternate)

        scale = 100

    elif args.model == "MiniONN_CNN_MNIST":
        model = models.MiniONN_CNN_MNIST(alternate=args.alternate)
        discrete_model = models.MiniONN_CNN_MNIST(alternate=args.alternate)

        scale = 100

    elif args.model == "MiniONN_CNN_CIFAR":
        model = models.MiniONN_CNN_CIFAR(pooling="max",alternate=args.alternate)
        discrete_model = models.MiniONN_CNN_CIFAR(pooling="max",alternate=args.alternate)

        scale = 100

    elif args.model == "DeepSecure_CNN_MNIST":
        model = models.DeepSecure_CNN_MNIST(alternate=args.alternate)
        discrete_model = models.DeepSecure_CNN_MNIST(alternate=args.alternate)

        scale = 100

    elif args.model == "test_MLP_MNIST":
        model = models.test_MLP_MNIST()
        discrete_model = models.test_MLP_MNIST()
        scale = 10
    else:
        print("How did you get here? Welcome!")

    x_train = dataset["x_train"]
    x_test = dataset["x_test"]

    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

 
    model.fit(x_train, y_train, validation_data=(x_test,y_test),
              epochs=epochs,batch_size=batch_size)


    model.save(savedir + args.model + "_" + str(int(time.time())) + ".h5")

    print("\nINFO: Evaluating model")
    evaluate(model,x_test,y_test,batch_size)

    weights = model.get_weights()

    discrete_weights = scale_round(weights,scale) # TODO: make this better
    discrete_model.set_weights(discrete_weights)

    print("\nINFO: Evaluating discretized model")
    evaluate(discrete_model,x_test,y_test,batch_size)

    print("\nINFO: Saving trained model files to {}".format(savedir))


    s.save_model(discrete_model,savedir)
    s.save_weights(discrete_weights,savedir)
    s.save_x_test(x_test.astype(int),savedir)
    s.save_y_test(y_test,savedir)
