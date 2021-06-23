# HappyMonk-AI-Assignment
PROGRAMMING TEST: LEARNING ACTIVATIONS IN NEURAL NETWORKS


I. BACKGROUND
Selection of the best performing AF for classification task is essentially a naive (or brute-force) procedure wherein, a popularly used AF is picked and used in the network for approximating the optimal function. If this function fails, the process is repeated with a different AF, till the network learns to approximate the ideal function. It is interesting to inquire and inspect whether there exists a possibility of building a framework which uses the inherent clues and insights from data and bring about the most suitable AF. The possibilities of such an approach could not only save significant time and effort for tuning the model, but will also open up new ways for discovering essential features of not-so-popular AFs.


II. PROBLEM STATEMENT
Given a specific activation function
        g(x) = k0 + k1x                                                                                                                                                                                                                           
and categorical cross-entropy loss, design a Neural Network on Banknote('BankNote_Authentication.csv') data where the activation function parameters k0, k1 are learned from the data you choose from one of the above-mentioned data sets. My solution was included by the learnable parameter values. I.e. final k0, k1 values at the end of training, A plot depicting changes in k0, k1 at each epoch, Training vs test loss, train vs. Test accuracy and a Loss function plot.


NOTE :
•	This report containing implementation details (al-gorithm, initial settings such as sampling the parameters k0, k1 from some distribution, parameter updates one pochs, final parameter values at the end of training, train vs test loss, train and test accuracy, F1-Score, plot of the loss function vs. epochs.
•	And Focusing on how to classify the detection technique of counterfeit banknotes. The approach that will be implemented to solve this Binary classification problem is by using the method of Artificial Neutral Network(ANN).

Data for this analysis Is :https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt

Part 1 : Data Preprocessing
variance	skewness	curtosis	entropy	class
0	3.62160	8.6661	-2.8073	-0.44699	0
1	4.54590	8.1674	-2.4586	-1.46210	0
2	3.86600	-2.6383	1.9242	0.10645	0
3	3.45660	9.5228	-4.0112	-3.59440	0
4	0.32924	-4.4552	4.5718	-0.98880	0 
In this dataset variance, skewness, curtosis, entropy are features
Whereas the class column contains the label 
Part 2 : Making a ANN classifier
Part 3 : Making the predictions and evaluting the model
# Making the confusion Matrix
              precision    recall  f1-score   support

           0       1.00      0.97      0.99       195
           1       0.97      1.00      0.98       148

    accuracy                           0.99       343
   macro avg       0.98      0.99      0.99       343
weighted avg       0.99      0.99      0.99       343
# accuracy: (tp + tn) / (p + n)
Accuracy score : 0.9854227405247813
# precision tp / (tp + fp)
Precision: 0.967320
# recall: tp / (tp + fn)
Recall: 1.000000
# f1: 2 tp / (2 tp + fp + fn)
F1 score: 0.983389
Part 4 Visualizing the results
From the plot of accuracy we can see that the model could probably be trained a little more as the trend for accuracy on both datasets.
# summarize history for accuracy
From the plot of loss, we can see that the model has comparable performance on both train and test dataset If these parallel plots start to depart consistently.
# summarize history for loss 
