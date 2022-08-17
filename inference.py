import os
from uNet import uNet
from dataGenerator import dataGenerator
import numpy as np
from tqdm import tqdm


def inference(img_shape, mcd, bayesian, data_dir, fpass):
    if mcd==True:
        model_path = 'MCD/models/'
        inf_path = 'MCD/inference/'
    if bayesian==True:
        model_path = 'Bayesian/models/'
        inf_path = 'Bayesian/inference/'
    else:
        model_path = 'Ensemble/models/'
        inf_path = 'Ensemble/inference/'
    if not os.path.exists(inf_path):
        os.makedirs(inf_path)

    data_dir_test_imgs = os.path.join(data_dir, 'testImgs/')
    data_dir_test_labels = os.path.join(data_dir, 'testLabels/')

    generator = dataGenerator(data_dir_test_imgs, data_dir_test_labels, batchSize=1)
    test_generator = generator.test(img_shape)

    model = uNet(img_shape[0], img_shape[1])
    if mcd:
        model = model.build(4, dropout_rate=0.5, mcd=mcd, bayesian = False)
    elif bayesian:
        model = model.build(4, mcd=False, bayesian = bayesian)
    else:
        model = model.build(4, dropout_rate=0.0, mcd=False, bayesian = False)
    # sampling for ensemble
    if not mcd and not bayesian:
        model.load_weights(model_path+'best_0.hdf5')
        preds = model.predict(test_generator)
        preds = preds[np.newaxis, ...]
        for i in tqdm(range(fpass-1)):
            model.load_weights(model_path+'best_{}'.format(i))
            pred = model.predict(test_generator)
            pred = pred[np.newaxis, ...]
            preds = np.append(preds, pred, axis=0)
    # sampling for mcd or bayesian
    else:
        model.load_weights(model_path+'best_mcd01.hdf5')
        preds = model.predict(test_generator)
        preds = preds[np.newaxis, ...]
        for i in tqdm(range(fpass-1)):
            pred = model.predict(test_generator)
            pred = pred[np.newaxis, ...]
            preds = np.append(preds, pred, axis=0)

    np.save(inf_path+'preds.npy', preds)
    return


def load_pred(inf_path):   
    return np.load(inf_path+'preds.npy')

