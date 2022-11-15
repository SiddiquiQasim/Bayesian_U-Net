import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback


def plot(iou, dice, mAP, loss, valiou, valdice, valmAP, valloss):
    plt.figure(figsize=(8,8))

    plt.subplot(2,2,1)
    plt.plot(iou)
    plt.plot(valiou)
    plt.title('model iou')
    plt.ylabel('iou')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(2,2,2)
    plt.plot(dice)
    plt.plot(valdice)
    plt.title('model dice')
    plt.ylabel('dice')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(2,2,3)
    plt.plot(mAP)
    plt.plot(valmAP)
    plt.title('model map')
    plt.ylabel('map')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(2,2,4)
    plt.plot(loss)
    plt.plot(valloss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.tight_layout()
    plt.savefig('eval.png')
    return



class TrainingCallback(Callback):
    def on_train_begin(self, logs=None):
        
        self.iou = []
        self.dice = []
        self.mAP = []
        self.lo = []
        self.valiou = []
        self.valdice = []
        self.valmAP = []
        self.vallo = []

    def on_epoch_end(self, epoch, logs=None):
        
        self.iou.append(logs['cal_iou'])
        self.dice.append(logs['cal_dice'])
        self.mAP.append(logs['cal_map'])
        self.lo.append(logs['loss'])
        self.valiou.append(logs['val_cal_iou'])
        self.valdice.append(logs['val_cal_dice'])
        self.valmAP.append(logs['val_cal_map'])
        self.vallo.append(logs['val_loss'])
        plot(self.iou, self.dice, self.mAP, self.lo, self.valiou, self.valdice, self.valmAP, self.vallo)