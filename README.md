# Deep Learning for COVID19 Detection from Lung Scans

We present a new opportunity to diagnose COVID computationally from lung scans. We used a publicly available dataset on Kaggle which contains a set of manually annotated lung CT scans with and without COVID-19. We built our predictive models using three well-known convolutional neural network architectures - AlexNet, VGG16, ResNet - and a custom architecture. Subsequently, we ensembled the models together for additional predictive power. 

Ultimately, the AlexNet model produced the best results with an accuracy of 84.38\%, F1 score of 85.23\%, and an AUC-ROC of 90.54%.

(Built by Santosh Murugan, Nicolai Ostberg, and Derek Jow).

---------------------------------------------------------------------------------------------------------
Usage: 

main.py is the program which calls other modules for functionality. It can be run with the following command-line flags: <br />

--model: Specifies which model to be run. Options are alex-net/resnet/vgg/custom. Default is alex-net.<br />
--epochs: Number of epochs to train the model. Default is 10.<br />
--cpu: Boolean for whether to use CPU instead of GPU. Default is False.<br />
