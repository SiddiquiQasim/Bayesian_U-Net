import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class dataGenerator:
    def __init__(self, dataDirImgs, dataDirLabels, batchSize=32, SEED=909):
        self.dataDirImgs = dataDirImgs
        self.dataDirLabels = dataDirLabels
        self.batchSize = batchSize
        self.SEED = SEED


    def segmentation(self,imageSize, color_mode='grayscale', data_gen_args = dict(rescale=1./255)):
        '''
        imageSize = (imageHeight, imageWidth)
        color_mode ='grayscale' if in_channels = 1
                   ='rgb' if in_channels = 3
                   ='rgba' if in_channels = 4
        data_gen_args = dict with arguments for augmenting data   '''
        
        img_datagen = ImageDataGenerator(**data_gen_args)
        label_datagen = ImageDataGenerator(**data_gen_args)
        
        img_generator = img_datagen.flow_from_directory(self.dataDirImgs, target_size=imageSize, class_mode=None, color_mode=color_mode, batch_size=self.batchSize, seed=self.SEED)
        label_generator = label_datagen.flow_from_directory(self.dataDirLabels, target_size=imageSize, class_mode=None, color_mode=color_mode, batch_size=self.batchSize, seed=self.SEED)
        return (pair for pair in zip(img_generator, label_generator))
    def testSegmentation(self,imageSize, color_mode='grayscale', data_gen_args = dict(rescale=1./255)):
        '''
        imageSize = (imageHeight, imageWidth)
        color_mode ='grayscale' if in_channels = 1
                   ='rgb' if in_channels = 3
                   ='rgba' if in_channels = 4
        data_gen_args = dict with arguments for augmenting data   '''
        
        img_datagen = ImageDataGenerator(**data_gen_args)
        label_datagen = ImageDataGenerator(**data_gen_args)
        
        img_generator = img_datagen.flow_from_directory(self.dataDirImgs, target_size=imageSize, class_mode=None, color_mode=color_mode, batch_size=self.batchSize, seed=self.SEED, shuffle=False)
        label_generator = label_datagen.flow_from_directory(self.dataDirLabels, target_size=imageSize, class_mode=None, color_mode=color_mode, batch_size=self.batchSize, seed=self.SEED, shuffle=False)
        return (pair for pair in zip(img_generator, label_generator))
    
    def test(self,imageSize, color_mode='grayscale', data_gen_args = dict(rescale=1./255)):
        img_datagen = ImageDataGenerator(**data_gen_args)
        img_generator = img_datagen.flow_from_directory(self.dataDirImgs, target_size=imageSize, class_mode=None, color_mode=color_mode, batch_size=self.batchSize, seed=self.SEED, shuffle=False)
        return img_generator

    def display(self, display_list):
        plt.figure(figsize=(12,12))
    
        title = ['Input Image', 'True Label', 'Predicted Label']
        
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='gray')
            
        plt.show()


    def visualizeDataset(self, datagen, model=None, num=1):
        for i in range(num):
            if model:
                image, label = next(datagen)
                pred_labels = model.predict(image)
                self.display([image[0], label[0], pred_labels[0]])
            else:
                image, label = next(datagen)
                self.display([image[0], label[0]])

    def load_pred(inf_path):   
        return np.load(inf_path+'preds.npy')


# if __name__ == '__main__':
#     dataDir = '/content/gdrive/MyDrive/Task02_Heart/Task02_Heart/slices/'
#     dataDirTrainImgs = os.path.join(dataDir, 'imgs/')
#     dataDirTrainLabels = os.path.join(dataDir, 'labels/')

#     dataDirTestImgs = os.path.join(dataDir, 'testImgs/')
#     dataDirTestLabels = os.path.join(dataDir, 'testLabels/')

#     generator = dataGenerator(dataDirTrainImgs, dataDirTrainLabels)
#     train_generator = generator.segmentation((320,320))

#     generator = dataGenerator(dataDirTestImgs, dataDirTestLabels)
#     test_generator = generator.segmentation((320,320))