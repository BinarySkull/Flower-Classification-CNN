# Flower Classification CNN

This is a project description for Udacity's AI Programming with Python Nanodegree program. The project involves developing an image classifier using PyTorch and converting it into a command-line application. 
The classifier is trained to recognize 102 different species of flowers using a dataset.

# How to use:
# Command line applications train.py and predict.py

Arguments for train.py: 

'data_dir'. 'Provide data directory. Mandatory argument', type = str
'--save_dir'. 'Provide saving directoryt', type = str
'--arch'. 'architecture (default \'vgg16\')', type = str
'--learning_rate'. 'set learning rate, type = float
'--hidden_units'. 'Hidden units in Classifier, type = int
'--epochs'. 'Number of epochs', type = int
'--gpu'. "Option to use GPU", type = str

Following arguments mandatory or optional for predict.py

'image_path'. 'Provide path to image. Mandatory argument', type = str
'checkpoint'. 'Provide path to checkpoint. Mandatory argument', type = str
'--top_k'. 'Return top k most likely classes', type = int
'--category_names'. 'Use a mapping of categories to real names', type = str
'--gpu'. "Use GPU for inference", type = str

#### Utils 1 and 2 contains important functions used in predict.py and train.py , I put them in separate files for the code to be cleaner and improve readability.

#### code can be found here: 
https://github.com/udacity/aipnd-project
