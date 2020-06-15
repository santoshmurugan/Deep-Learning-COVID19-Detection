import torch
import torchvision.models as models
import numpy as np
import argparse
from trainer import Trainer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from dataloader import get_dataloaders
import custom
import ensemble
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="alex-net", type=str, help='alex-net/resnet/vgg/custom')
    parser.add_argument('--load_from_file', default="", type=str, help='relative path of model file to load')
    parser.add_argument('--epochs', default=3, type=int, help='number of epochs')
    parser.add_argument('--cpu', default=False, type=bool, help='use CPU instead of GPU')
    args = parser.parse_args()
    batch_size = 1
    model = args.model
    epochs = args.epochs
    model_path = args.load_from_file
    use_cpu = args.cpu

    if args.cpu:
        print("Using the CPU")
        device = torch.device("cpu")
    else: 
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("There are %d GPU(s) available." % torch.cuda.device_count())
            print("We will use the GPU: ", torch.cuda.get_device_name(0))
        else:
            print("No GPU available, using the CPU instead")
            device = torch.device("cpu")

    tvmodel = None
    if model == "alex-net":
        tvmodel = models.alexnet(pretrained=True)
    elif model == "vgg":
        tvmodel = models.vgg11_bn(pretrained=True)
    elif model == 'resnet':
        tvmodel = models.resnet18(pretrained = True)
    elif model == "custom":
        tvmodel = custom.NovelNet()
    elif model == "ensemble":
        tvmodel = ensemble.EnsembleModel()
    else:
        print("Incorrect model was passed, exiting!")
        exit()

    print("Loading data...")
    train_dataloader, test_dataloader = get_dataloaders(device) # torchtensors
    print("Done Loading Data.")
   
    trainer = Trainer(epochs=epochs, batch_size=batch_size, learning_rate=1e-5, model=tvmodel, model_name = model, device=device)

    if model_path != "":
        print("Loading Model")
        trainer.model = torch.load(model_path)
        trainer.model.eval()
        print("Finished Loading Model")
    else:
        print("Fitting model...")
        trainer.fit(train_dataloader) 
        print("Done Fitting Model")

    prediction, probs = trainer.predict(test_dataloader)
    prediction = np.array(prediction)

    probs = torch.cat(probs, dim=0)
    probs = np.array(probs)

    accuracy = accuracy_score(test_dataloader[:][1], prediction)
    print("Test accuracy: %.4f" % accuracy)

    f1 = f1_score(test_dataloader[:][1], prediction)
    print("Test F1: %.4f" % f1)

    auroc = roc_auc_score(test_dataloader[:][1], prediction)
    print("Test AUC_ROC: %.4f" % auroc)

    precision = precision_score(test_dataloader[:][1], prediction)
    print("Test Precision: %.4f" %  precision)

    recall = recall_score(test_dataloader[:][1], prediction)
    print("Test Recall: %.4f" % recall)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(probs.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(test_dataloader[:][1], probs[:, i])
        roc_auc[i] = roc_auc_score(test_dataloader[:][1], probs[:, i])

    plt.figure()
    #plt.plot(fpr[0], tpr[0], color='red', label='ROC curve (area = %0.4f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc[1])
    plt.plot([0,1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for Alex Net')
    plt.legend(loc="lower right")
    plt.savefig("alex-net-roc.png")

if __name__ == "__main__":
    main()