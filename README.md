
# Education Project Cat vs. Dog Image Classifier

Overview
This repository contains code to create a deep learning image classifier that can distinguish between images of cats and dogs. The classifier is built using TensorFlow and Keras and uses a Convolutional Neural Network (CNN) architecture. It is trained on the Dogs vs. Cats dataset obtained from Kaggle.




## Table of Contents
Setup


## Setup
Kaggle API Setup
Before running the code, you need to set up the Kaggle API to download the dataset. Follow these steps:

1. Upload your Kaggle API key (kaggle.json) using the provided code:
   
   from google.colab import files
   
   ! pip install -q kaggle

files.upload()

!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!rm kaggle.json

2. Download the Dogs vs. Cats dataset:
   
   !kaggle datasets download -d salader/dogs-vs-cats
3. Extract the dataset:
   
   import zipfile

zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')

zip_ref.extractall('/content')

zip_ref.close()

Python Environment

4. You can install the required Python packages using the following command:

pip install -r requirements.txt
## DataSet
The model is trained on the Dogs vs. Cats dataset, which contains a large number of cat and dog images. The dataset is divided into training and testing sets and is organized in the /content/train and /content/test directories.
## DataSet
The model is trained on the Dogs vs. Cats dataset, which contains a large number of cat and dog images. The dataset is divided into training and testing sets and is organized in the /content/train and /content/test directories.
## Model Architeture
Model Architecture
The CNN model is defined using Keras and has the following architecture:


Input Layer (256x256 pixels, 3 channels)

Convolutional Layer (32 filters, kernel size 3x3)

Batch Normalization

Max Pooling Layer (2x2)

Convolutional Layer (64 filters, kernel size 3x3)

Batch Normalization

Max Pooling Layer (2x2)

Convolutional Layer (128 filters, kernel size 3x3)

Batch Normalization

Max Pooling Layer (2x2)

Flatten Layer

Dense Layer (128 units, ReLU activation)

Dropout Layer (Dropout rate: 0.1)

Dense Layer (64 units, ReLU activation)

Dropout Layer (Dropout rate: 0.1)

Output Layer (1 unit, Sigmoid activation)
## Training
The model is trained using the training dataset with a batch size of 32 and for 10 epochs. It is compiled using the Adam optimizer and binary cross-entropy loss.

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
## Evaluation
Model performance can be evaluated using the test dataset. Metrics such as accuracy, loss, precision, recall, and F1-score can be calculated. You can also visualize training and validation performance using Matplotlib.
## Testing
To test the model on custom images, you can load an image, preprocess it (resize to 256x256 pixels), and then use the model for prediction.

import cv2

test_img = cv2.imread('/content/dog.jpeg')

test_img = cv2.resize(test_img, (256, 256))

test_input = test_img.reshape((1, 256, 256, 3))

model.predict(test_input)
## Acknowledgements

Would like to express my gratitude to the following following contributors and resources that have played a significant role in the development of this project:

- **[Campus X YouTube Channel](https://www.youtube.com/watch?v=0K4J_PTgysc)**: Your   insightful tutorials and educational content provided by Campus X  greatly aided my understanding of deep learning and image classification.

- **Kaggle Dataset - Dogs vs. Cats**: Would like to thank the Kaggle community for providing the "Dogs vs. Cats" dataset, which has served as the foundation for our training data. 

I appreciate the open-source community and the invaluable knowledge shared by developers worldwide.

