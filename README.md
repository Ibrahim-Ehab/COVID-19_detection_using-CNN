# COVID-19_detection_using-CNN
COVID-19 Classifier using CNN DenseNet model 

 
 
# 1. Abstract
The need to streamline patient management for COVID-19 has become more pressing than ever. Chest X-rays provide a non-invasive (potentially bedside) tool to monitor the progression of the disease. 
In this study, we will create a convolutional neural network model classifier that can detect people with COVID-19 using the DenseNet layer and train it using chest x-ray images collected from chest x-rays (pneumonia) and for the general COVID-19 x-ray data collection. After training and evaluating the model, we got an accuracy of 97.6%, which means that out of every 10 people, there are three who are misdiagnosed.
#2. Introduction
In this study, we go through X-ray chest images to try to determine people’s status with COVID-19 and if it is positive or negative by building the CNN model. 
We select X-ray images because it is cheaper and take a short time rather than blood tests, could measure the spread of the virus, datasets are available, and more. 
Using the Convolutional Neural Networks DenseNet model, we will apply binary classification to X-ray images to build a model that can classify images in positive or negative COVID-19 status. This can limit the growth of the number of infected people.
Before going through the paper, we will illustrate some basic definitions.
AI: The ability of a digital computer or computer-controlled robot to perform tasks commonly associated with intelligent beings.
Machine Learning:  It’s a type of AI that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so.
Deep Learning: Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers. These neural networks attempt to simulate the behavior of the human brain.
Data mining: It’s the process of sorting through large data sets to identify patterns and relationships that can help solve business problems through data analysis.
 
#3. Related Work
 3.1-According to Classification using CNN
In this study, we will use the DenseNet layer to make our classification, but there are many other layers that could be used like VGG16, ResNet50, MobileNet, Inception V3, Xception, or NASNet. All of these do the same task but with different accuracy levels.
3.2-According to using the CNN DenseNet model
There are a lot of studies that use the DenseNet model not only to detect COVID-19 but also to predict the severity of COVID-19, to see the extent of the patient's improvement and the effectiveness of the drug. Other studies use it to locate the affected areas in the chest.
http://arxiv.org/abs/2005.11856 
https://arxiv.org/abs/2005.10052 
https://arxiv.org/abs/2006.04603 
#4. Materials and Methods
4.1-COVID-19 Cohort
We used a cohort of 930 posteroanterior (PA) CXR images from a public COVID-19 image data collection [ieee8023/covid-chestxray-dataset] for positive patients. 
All patients were reported COVID-19 positive and sourced from many hospitals around the world from December 2019 to March 2020+ COVID-19.
 for normal people (or negative patients) we used a cohort of 3340 posteroanterior (PA) CXR images from a Kaggle public dataset [Chest X-Ray Images (Pneumonia)] (but we actually take 930 images) - COVID-19.
4.2-Dataset preparation
We split images after rearranging them randomly into two sections, Train (which contains 700 images for two labels positive/negative COVID-19) for the training model
Test (which contains 230 images for two labels positive/negative COVID-19) for the test model and evaluate it. You can get a prepared dataset here.
4.3-Model architecture, Building model, and Preprocessing images
We do some processing on all images of the dataset like rescale, zoom range, and horizontal flip to pass them to our model.  
Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion, we used the DenseNet model which consists of 15 layers as follows: one-input layer, four-Convolutional layers, three-Pooling layers, four-Dropout layers, one-flatten layer, and two-DenseNet layers
4.4-CNN Architecture from scratch
 
 
one-input layer, four-Convolutional layers, three-Pooling layers, four-Dropout layers, one-flatten layer, and two-DenseNet layers
input layer: get processed images in new size (224,224,3)
Conv2D layer: has 32 filters with kernel size 3x3, to extract small features multiplied by the Input layer
Note that all the following layers will be multiplied by the variable X which used to store all architecture
Conv2D layer: has 64 filters with kernel size 3x3, to extract bigger than previous
MaxPooling2D and Dropout: reduce the number of neurons and extract important features and avoid overfitting 
Conv2D layer: has 64 filters with kernel size 3x3, to extract bigger features
MaxPooling2D and Dropout: reduce the number of neurons and extract important features and avoid overfitting
Conv2D layer: has 128 filters with kernel size 3x3, to extract bigger than previous
MaxPooling2D and Dropout: reduce the number of neurons and extract important features and avoid overfitting
Flatten: convert from matrix to column 
Dense: contain 64 neurons.
Dropout: but with a different value from others (0.5)
Dense: for binary classification using sigmoid function as activation function.
5. Training and Validation model
5.1-From scratch model
 
 
#5.2-Pretrained model 
 
 
#6. Classification report and Confusion Matrix
6.1-From scratch model
 
 
6.2-Pretrained model

 
 
#7. From Scratch VS pretrained models
When we run two models on same dataset, perform same processing on images, same batch size and number of epochs we get: 
97.6% VS 90.5% 
We notice that accuracy in these conditions be highest in from scratch model.
7. Future Work
we will upgrade the purpose of study by trying to predict the severity of COVID-19 and locate the affected areas in the chest. Compare all CNN models to determine which one is the best for such a use case. 
This study will help to develop a tool that could monitor the progression of the disease without surgery.







8. References 
https://arxiv.org/abs/1608.06993v5 
https://github.com/ieee8023/covid-chestxray-dataset 
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 
https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/ 
https://poloclub.github.io/cnn-explainer/ 
https://www.youtube.com/watch?v=nHQDDAAzIsI&t=2358s 
https://www.youtube.com/watch?v=Lz2D3Xg6uQY 
https://www.youtube.com/watch?v=-W6y8xnd--U 
https://analyticsindiamag.com/a-complete-understanding-of-dense-layers-in-neural-networks/ 
https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet/DenseNet201 
