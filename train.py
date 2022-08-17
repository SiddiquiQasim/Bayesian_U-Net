import os
from uNet import uNet
from dataGenerator import dataGenerator
from Loss import crossEntropyLoss
from Metrics import ioU, dice, mAP
import tensorflow as tf
from eval_plot import TrainingCallback
from argparse import ArgumentParser

def train(img_shape, mcd, bayesian, data_dir, num_of_epochs=100):
    data_dir_train_imgs = os.path.join(data_dir, 'imgs/')
    data_dir_train_labels = os.path.join(data_dir, 'labels/')
    data_dir_test_imgs = os.path.join(data_dir, 'testImgs/')
    data_dir_test_labels = os.path.join(data_dir, 'testLabels/')

    if mcd==True:
        model_path = 'MCD/models/'
    if bayesian==True:
        model_path = 'Bayesian/models/'
    else:
        model_path = 'Ensemble/models/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    generator = dataGenerator(data_dir_train_imgs, data_dir_train_labels)
    train_generator = generator.segmentation(img_shape)
    generator = dataGenerator(data_dir_test_imgs, data_dir_test_labels)
    test_generator = generator.segmentation(img_shape)

    model = uNet(img_shape[0], img_shape[1])
    if mcd:
        model = model.build(4, dropout_rate=0.5, mcd=mcd, bayesian = False)
    elif bayesian:
        model = model.build(4, mcd=False, bayesian = bayesian)
    else:
        model = model.build(4, dropout_rate=0.0, mcd=False, bayesian = False)

    iou = ioU()
    dice = dice()
    mAP = mAP(0.5)
    loss = crossEntropyLoss()

    model.compile(optimizer="adam", loss=loss, metrics=[iou, dice, mAP])
    
    if not mcd and not bayesian:
        saveBest = tf.keras.callbacks.ModelCheckpoint(model_path + 'best_{}.hdf5'.format(idx), save_weights_only=True,
                                                    save_best_only=True, monitor='loss', mode='min',
                                                    verbose = 0)
        if os.path.exists(model_path+'best_{}.hdf5'.format(idx)):
            model.load_weights(model_path+'best_{}.hdf5'.format(idx))
        history  = model.fit_generator(generator=train_generator, validation_data=test_generator,
                            epochs=num_of_epochs, callbacks=[saveBest, TrainingCallback()])
        model.save_weights(model_path+'last_{}.hdf5'.format(idx))
    else:
        saveBest = tf.keras.callbacks.ModelCheckpoint(model_path + 'best.hdf5', save_weights_only=True,
                                                        save_best_only=True, monitor='loss', mode='min',
                                                        verbose = 0)
        if os.path.exists(model_path+'best.hdf5'):
            model.load_weights(model_path+'best.hdf5')
        history  = model.fit_generator(generator=train_generator, validation_data=test_generator,
                            epochs=num_of_epochs, callbacks=[saveBest, TrainingCallback()])
        model.save_weights(model_path+'last.hdf5')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("index", type=int, help="Save model index")
    args = parser.parse_args()
    idx = args.index

    train(img_shape=(320,320), mcd=False, bayesian=False, data_dir='data/slices/', num_of_epochs=100)