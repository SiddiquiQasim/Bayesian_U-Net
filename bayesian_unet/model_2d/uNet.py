import tensorflow as tf
import tensorflow_probability as tfp


class uNet:

    def __init__(self, img_height, img_width, in_channels=1, out_channels=1):
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.out_channels = out_channels

    def build(self, n_levels, initial_feature=32, n_blocks=2, kernel_size=3, pooling_size=2, dropout_rate=0.0, mcd=False, bayesian = False):
        inputs = tf.keras.layers.Input(shape=(self.img_height, self.img_width, self.in_channels))
        x = inputs
        convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')
        #downstream
        skips ={}
        for level in range(n_levels):                    
            for _ in range(n_blocks):
                if bayesian:
                    x = tfp.layers.Convolution2DFlipout(initial_feature * (2 ** level), **convpars)(x)
                else:
                    x = tf.keras.layers.Conv2D(initial_feature * (2 ** level), **convpars)(x)
            if not bayesian:
                x = tf.keras.layers.Dropout(dropout_rate)(x, training = mcd)
            if level < n_levels - 1:
                skips[level] = x
                x = tf.keras.layers.MaxPool2D(pooling_size)(x)
        #upstream
        for level in reversed(range(n_levels-1)):
            x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
            x = tf.keras.layers.Concatenate()([x, skips[level]])
            for _ in range(n_blocks):
                if bayesian:
                    x = tfp.layers.Convolution2DFlipout(initial_feature * (2 ** level), **convpars)(x)
                else:
                    x = tf.keras.layers.Conv2D(initial_feature * (2 ** level), **convpars)(x)
            if not bayesian:
                    x = tf.keras.layers.Dropout(dropout_rate)(x, training = mcd)
        #output        
        if bayesian:
            outputs = tfp.layers.Convolution2DFlipout(self.out_channels, kernel_size=1, activation='sigmoid', padding='same')(x)
            return tf.keras.Model(inputs=[inputs], outputs=[outputs], name=f'UNET-L{n_levels}-F{initial_feature}-Flipout_CE')
        else:
            outputs = tf.keras.layers.Conv2D(self.out_channels, kernel_size=1, activation='sigmoid', padding='same')(x)
            return tf.keras.Model(inputs=[inputs], outputs=[outputs], name=f'UNET-L{n_levels}-F{initial_feature}')


if __name__ == '__main__':
    unet = uNet(320, 320)



