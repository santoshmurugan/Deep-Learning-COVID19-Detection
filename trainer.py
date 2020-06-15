import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pathlib import Path
import numpy as np

class Trainer:

    def __init__(self, epochs, batch_size, learning_rate, model, model_name, device):
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = model
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Following this: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        # forward pass
        if self.model_name == 'alex-net':
            self.model.classifier[6] = nn.Linear(4096,2)
        elif self.model_name == 'resnet':
            self.model.fc = nn.Linear(512,2)
        elif self.model_name == 'vgg':
            self.model.classifier[6] = nn.Linear(4096,2)

        self.model.to(device)
        self.device = device

    def fit(self, dataset):
        for epoch in range(self.num_epochs):
            print("Beginning EPOCH %d!" % epoch)

            loss_func = nn.BCEWithLogitsLoss()

            print("Beginning Training!")
            for i, data in tqdm(enumerate(dataset), total=len(dataset)):
                
                # xi = (features, label)
                x, y = data
                y_expected = torch.tensor([y==0, y==1], device=self.device, dtype=torch.float)

                # sgd -> Batch size of 1
                x = torch.unsqueeze(x, dim=0)

                # set gradients to 0
                self.optimizer.zero_grad()

                output = self.model(x)
                output = output.squeeze()

                loss = loss_func(output, y_expected)

                loss.backward()
                self.optimizer.step()
                # print update messages
                if i % 100 == 0:
                    print("Iter: %d, Loss: %.4f" % (i, loss))

            print("Finished EPOCH %d!" % epoch)
            self.save(epoch, loss, self.model_name, '.model')

            # Compute current accuracy, f1, etc.
            preds, _ = self.predict(dataset)
            accuracy = accuracy_score(dataset[:][1], preds)
            print("Training accuracy: %.4f" % accuracy)


    def predict(self, dataset):
        print("Computing predictions....")
        predictions = []
        probs = []
        for i, data in tqdm(enumerate(dataset), total=len(dataset)):
            x, y = data
            x = torch.unsqueeze(x, dim=0)

            with torch.no_grad():
                output = self.model(x)
            
            result = np.argmax(output)

            predictions.append(result)
            probs.append(output)
        return predictions, probs

    def save(self, epoch, loss, model_prefix='model', root='.model'):
        print("Saving model...")
        path = Path(root) / (model_prefix + '.ep%d' % epoch)
        if not path.parent.exists():
            path.parent.mkdir()
        
        torch.save(self.model, path)
        #torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'loss': loss}, path)
