
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from dataGenerator import dataGenerator
from inference import load_pred


def inference(img_shape, mcd, bayesian, data_dir, idx):
    if mcd==True:
        output_img = 'MCD/uncertainty_vis/'
        inf_path = 'MCD/inference/'
    if bayesian==True:
        output_img = 'Bayesian/uncertainty_vis/'
        inf_path = 'Bayesian/inference/'
    else:
        output_img = 'Ensemble/uncertainty_vis/'
        inf_path = 'Ensemble/inference/'
    if not os.path.exists(inf_path):
        os.makedirs(inf_path)

    data_dir_test_imgs = os.path.join(data_dir, 'testImgs/')
    data_dir_test_labels = os.path.join(data_dir, 'testLabels/')

    preds = load_pred(inf_path)
    pred_mean = np.mean(preds, axis=0, keepdims=True)
    pred_std = np.std(preds, axis=0, keepdims=True)

    generator = dataGenerator(data_dir_test_imgs, data_dir_test_labels, batchSize=1)
    test_generator = generator.test(img_shape)
    label_generator =generator.testSegmentation(img_shape)

    title = ['Input', 'True Label', 'Pred Label', 'Uncertainty']
    display_list = [test_generator[idx][0], label_generator[idx][0], pred_mean[0][idx], pred_std[0][idx]]
    for j in range(len(display_list)):
        plt.subplot(1, len(display_list), j+1)
        plt.title(title[j])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[j]), cmap='gray')   
    plt.savefig(output_img+'uncertainty_img-{}.png'.format(idx))
    return