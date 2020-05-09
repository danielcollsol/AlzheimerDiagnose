#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_cell_magic('html', '', '\n<!-- Just run this cell to provide style to the notebook-->\n<!-- This is a code to add style to the notebook, it is based on a .css on GitHub http://bit.ly/1Bf5Hft -->\n\n<style>\n\nhtml {\n  font-size: 62.5% !important; }\nbody {\n  font-size: 1.5em !important; /* currently ems cause chrome bug misinterpreting rems on body element */\n  line-height: 1.6 !important;\n  font-weight: 400 !important;\n  font-family: "Raleway", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif !important;\n  color: #222 !important; }\n\ndiv{ border-radius: 0px !important;  }\ndiv.CodeMirror-sizer{ background: rgb(244, 244, 248) !important; }\ndiv.input_area{ background: rgb(244, 244, 248) !important; }\n\ndiv.out_prompt_overlay:hover{ background: rgb(244, 244, 248) !important; }\ndiv.input_prompt:hover{ background: rgb(244, 244, 248) !important; }\n\nh1, h2, h3, h4, h5, h6 {\n  color: #333 !important;\n  margin-top: 0 !important;\n  margin-bottom: 2rem !important;\n  font-weight: 300 !important;\n    text-decoration: underline;\n}\nh1 { font-size: 4.0rem !important; line-height: 1.2 !important;  letter-spacing: -.1rem !important;}\nh2 { font-size: 3.6rem !important; line-height: 1.25 !important; letter-spacing: -.1rem !important; }\nh3 { font-size: 3.0rem !important; line-height: 1.3 !important;  letter-spacing: -.1rem !important; }\nh4 { font-size: 2.4rem !important; line-height: 1.35 !important; letter-spacing: -.08rem !important; }\nh5 { font-size: 1.8rem !important; line-height: 1.5 !important;  letter-spacing: -.05rem !important; }\nh6 { font-size: 1.5rem !important; line-height: 1.6 !important;  letter-spacing: 0 !important; }\n    \n@media (min-width: 550px) {\n  h1 { font-size: 5.0rem !important; }\n  h2 { font-size: 4.2rem !important; }\n  h3 { font-size: 3.6rem !important; }\n  h4 { font-size: 3.0rem !important; }\n  h5 { font-size: 2.4rem !important; }\n  h6 { font-size: 1.5rem !important; }\n}\n\np {\n    margin-top: 0 !important;\n    margin-bottom: 1rem !important;\n    text-align: justify;\n    text-justify: inter-word;\n    line-height: 1.5 !important;\n    font-size: 1.2em !important;\n    font-family: "Raleway", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif !important;\n}\n  \na {\n  color: #1EAEDB !important; }\na:hover {\n  color: #0FA0CE !important; }\n  \ncode {\n  padding: .2rem .5rem !important;\n  margin: 0 .2rem !important;\n  font-size: 90% !important;\n  white-space: nowrap !important;\n  background: #F1F1F1 !important;\n  border: 1px solid #E1E1E1 !important;\n  border-radius: 4px !important; }\npre > code {\n  display: block !important;\n  padding: 1rem 1.5rem !important;\n  white-space: pre !important; }\n  \nbutton{ border-radius: 0px !important; }\n.navbar-inner{ background-image: none !important;  }\nselect, textarea{ border-radius: 0px !important; }\n    \n#Top_Header {\n    background-size: contain;\n    background-repeat: no-repeat;\n    background-position: left, right;\n}\n\n.output {\n    display: flex;\n    align-items: center;\n    text-align: center;\n}\n    \n</style>\n    \n<script> \n$( document ).ready(function () {\n    $("div#notebook-container").children().first().hide();\n});\n</script>')


# In[2]:


get_ipython().run_cell_magic('javascript', '', '//Execute this code to show the styling of the document\n$("div#notebook-container").children().first().show();')


# 
# <div id="Top_Header">
#     <center>
#         <h1>Alzheimer Disease Identification</h1>
#         <h3>Deep Learning: Transfer Learning</h3>
#         <h6>Daniel Coll, TFM 2019</h6>
#     </center>
# </div>

# <div id="Introduction">
#     <h2>Introduction</h2>
#     <p>This notebook proposes an approach to classify Structural Magnetic Resonance Imaging (MRI) for Alzheimer Disease identification using <i>Convolutional Neural Networks</i> and <i> Transfer Learning</i>.</p>
#     <p><i> Transfer Learning</i> is an excellence approach for problems where there isn't a large amount of data. Using pre-trained models helps us increase our accuracy, since these models are trained on datasets with millions of images. Adding our own fully connected layers to adapt these models to this particular problem is the final step for this procedure. </p>
# </div>

# <h2> Importing </h2>

# In[4]:


import pandas as pd
import numpy as np
import os
import tensorflow as tf 
import keras 
import matplotlib.pyplot as plt
import pickle
import sklearn as skl
import itertools
from keras import regularizers
from PIL import Image
from sklearn.metrics import confusion_matrix
from PIL import Image
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,BatchNormalization
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers     
from keras.applications.inception_v3 import preprocess_input


# <h1> Auxiliar Functions </h1>

# <p> The following cell includes some useful functions.  </p>

# <h1>Path and Labels </h1>

# <p> Labels are stored in the list named "classes" and are ordered according to the structure in the dataset folder where the images are stored. </p>
# <p>The dataset root paths for the training, validation and test sets are stored in four different dictionaries. </p>

# In[5]:


# Classes & datasets
classes = ['AD',  'CN'] 
datasets = ['MPRAGE_2Cat', 'MPRAGE_2CatRed']


# In[6]:


# Dataset's Paths
for dataset in datasets:
    vars()['dataset_{}_path'.format(dataset)] = {'train': '../data/Datasets/{}/TrainingSet/'.format(dataset,dataset),
                 'validation': '../data/Datasets/{}/CrossValSet/'.format(dataset,dataset), 
                 'test':'../data/Datasets/{}/TestSet/'.format(dataset,dataset) }
    
    
print('Path to training dataset 1: ', dataset_MPRAGE_2Cat_path.get('train'))
print('Path to validation dataset 2: ', dataset_MPRAGE_2CatRed_path.get('validation'))
print('Path to test dataset 3: ', dataset_MPRAGE_2CatRed_path.get('test'))


#  <h1>Neural Network Architecture </h1>

# <p>In Keras, pre-trained models are stored in <i>keras.applications</i> and are treated as layers when designing a network. As we call <i>model.add(Convolution(...)</i>, we would call <i>model.add(Xception(...))</i>. Paremeters for that method are: weights, include_top and input_shape. </p>
#     
# <p> We can decide whether to use the pretrained weights by calling <i>weights = 'imagenet' </i>  or just stick with the structure and randomly initialize them. Include_top makes reference to include the fully connected layers in the model or not. Usually, in transfer learning, these layers are not included because the output layer differs and we are predicting new classes. Finally, input_shape is the dimension of your images in the classic format [height, width, channels]. There are default dimensions in which the models were trained, but any dimension can be used. The layer would automatically infer changes on the convolutions, poolings etc...   </p>
# 
# <p> We opted for using default dimensions for each model. First of all, we define a function that returns the model we will use. Once we get the model, since the output shape is a 3D tensor we flatten it. Then, an undefined number of fully connected layers are added, each one with dropout, batch normalization and leaky relu as activation function. The output layer is the final one, and since this is classification problem we use softmax.  </p>

#  <h2>Importing pre-trained model and adding dense layers. </h2>

# <p> In this section we import InceptionV3 model without its last layer and add some dense layers and a softmax layer.</p>

# In[71]:


base_model=keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(255,255,3))
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(256,activation='relu')(x)
x = Dropout(0.25)(x)
preds=Dense(2,activation='softmax',
            kernel_regularizer=regularizers.l2(0.05),
            activity_regularizer=regularizers.l1(0.05))(x) 


# In[72]:


len(base_model.layers)


# In[73]:


base_model.summary()


# <p> Next we reduce the complexity of the model based removing last layers.</p>

# In[74]:


def remove_layers(num_layers):
    for iterations in list(range(num_layers-1)):
        base_model.layers.pop()


# In[75]:


#remove_layers() 


# In[76]:


len(base_model.layers)


# In[77]:


base_model.summary()


# <p> Next we make a model based on the architecture we have provided for every dataset.</p>

# In[78]:


model_inception3 = Model(inputs=base_model.input,outputs=preds)


# In[79]:


model_inception3.summary()


# <p1> Since we do not have a large dataset, we will set all the base model's parameters as non trainable and use the pre-trained network weights as initialisers </p1>

# In[80]:


def set_nontrainable_layers(models,num_layers):
    for model in models:
        for layer in model.layers[:num_layers]:
            layer.trainable=False
        for layer in model.layers[num_layers:]:
            layer.trainable=True


# In[81]:


set_nontrainable_layers([model_inception3],len(base_model.layers))


# <h2> Loading train and validation data into ImageDataGenerators</h2>

# In[82]:


datagen = ImageDataGenerator(preprocessing_function= preprocess_input, zca_whitening=True) 
for dataset in datasets:
    
    dataset_path = vars()['dataset_{}_path'.format(dataset)]
    print(dataset_path)
    vars()['train_generator_{}'.format(dataset)]=datagen.flow_from_directory(
                                                     dataset_path.get('train'), 
                                                     target_size=(255,255), 
                                                     color_mode='rgb',
                                                     batch_size=64,
                                                     class_mode='categorical', 
                                                     shuffle=True)
    vars()['val_generator_{}'.format(dataset)]=datagen.flow_from_directory(
                                                     dataset_path.get('validation'), 
                                                     target_size=(255,255), 
                                                     color_mode='rgb',
                                                     batch_size=64,
                                                     class_mode='categorical', 
                                                     shuffle=True) 
    
    vars()['test_generator_{}'.format(dataset)]=datagen.flow_from_directory(
                                                     dataset_path.get('test'), 
                                                     target_size=(255,255), 
                                                     color_mode='rgb',
                                                     batch_size=64,
                                                     class_mode='categorical', 
                                                     shuffle=True)
  


# <h2> Training and Evaluating model</h2>

# <p1> We compile the models using adam as optimizer, categorical cross entropy as loss function and accuracy as evaluation metric </p1>

# In[83]:


opt = keras.optimizers.RMSprop(lr=0.0001)

model_inception3.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy']) 


# In[65]:


# # Dataset MPRAGE_2CatRed
# step_size_train=train_generator_MPRAGE_2CatRed.n//train_generator_MPRAGE_2CatRed.batch_size
# print(step_size_train)
# history = model_VGG16.fit_generator(generator=train_generator_MPRAGE_2CatRed, validation_data=val_generator_MPRAGE_2CatRed, validation_steps=10,
#                    steps_per_epoch=step_size_train, epochs=5, verbose=1)


# In[84]:


# Dataset MPRAGE_2Cat
step_size_train=32 
print(step_size_train)
history = model_inception3.fit_generator(generator=train_generator_MPRAGE_2Cat, validation_data=val_generator_MPRAGE_2Cat,
                                    validation_steps=10,steps_per_epoch=step_size_train, epochs=100)


# ## Training Curve

# In[87]:


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'][:25])
plt.plot(history.history['val_acc'][:25])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'][:25])
plt.plot(history.history['val_loss'][:25])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
print(history.history)


# In[88]:


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# ## Singular prediction

# In[89]:


#Get image from to predict Post_Processada_2Cat dataset
img = Image.open('../data/Datasets/MPRAGE_2Cat/TestSet/AD/AD_51-frame000-slice119.jpg')
img = np.asarray(img.resize((255,255), Image.ANTIALIAS))/255
plt.imshow(img)
p = model_inception3.predict(np.array([img]))
p


# In[90]:


#Get image from to predict Post_Processada_2Cat dataset
img = Image.open('../data/Datasets/MPRAGE_2Cat/TestSet/AD/AD_51-frame000-slice127.jpg')
img = np.asarray(img.resize((255,255), Image.ANTIALIAS))/255
plt.imshow(img)
p = model_inception3.predict(np.array([img]))
p


# ## Multiple prediction

# In[91]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[92]:


# Get images and corresponding label
images_test, y_labels = next(test_generator_MPRAGE_2Cat)
images_test = np.round(images_test)
# Get predictions
y_pred = model_inception3.predict_generator(test_generator_MPRAGE_2Cat, steps=1, verbose=0)
y_pred = np.round(y_pred).argmax(axis=1)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_labels.argmax(axis=1), y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[ ]:




