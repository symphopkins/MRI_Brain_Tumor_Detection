# Brain Tumor Detection with CNN Model and FastAPI

## Overview

Used EfficientNet model to predict brain tumor types from MRI images, achieving a weighted average F1 score of 0.76.

## Dataset

The dataset used for training and testing the model is sourced from Kaggle. It can be found at the following link: [Kaggle Dataset: Brain Tumor MRI Classification]([https://www.kaggle.com/code/jaykumar1607/brain-tumor-mri-classification-tensorflow-cnn](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri ))

## Files

- **requirements.txt**: Text file listing all Python dependencies for reproducing the environment.
- **mri_model_training.ipynb**: Notebook containing the code for training the CNN model.
- **model.h5**: Trained CNN model saved in HDF5 format.
- **label_encoder.npy**: Numpy file containing label encoding information.
- **mri_model_api.py**: Python script for the FastAPI-based model API.
- **index.html**: HTML file for a simple interface to upload images and get predictions (optional).
- **glioma_tumor.jpeg**: Test image.

## Usage

1. **Training the Model**: Execute the code in `mri_model_training.ipynb` to train the CNN model using the provided dataset. Adjust hyperparameters as needed.

2. **Starting the API**: Run `mri_model_api.py` to start the FastAPI-based model API. Ensure that all dependencies listed in `requirements.txt` are installed.

3. **Testing the API**: Use any HTTP client to interact with the model API.

## License

MIT License

![API Screenshot](https://github.com/symphopkins/MRI_Brain_Tumor_Detection/blob/master/api.jpeg)

