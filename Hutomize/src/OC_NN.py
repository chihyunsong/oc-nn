import sys,os
import numpy as np
PROJECT_DIR = '/tf/Hutomize'
sys.path.append(PROJECT_DIR)
from KerasDataset import SurgicalDataset
import tensorflow as tf
from keras.layers import GaussianNoise, Input, Conv2D, UpSampling2D, BatchNormalization, ZeroPadding2D, MaxPooling2D
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from custom_layers.unpooling_layer import Unpooling
from utils import custom_loss
from keras.optimizers import SGD
from keras.utils.training_utils import multi_gpu_model


class OCNN(object):

    channels = 3
    image_height = 228
    image_width = 228

    def __init__(self):
        self.data = SurgicalDataset('data')
        self.generator = self.data.getIterator()
        self.ae = self.create_autoencoder()
        self.ae = multi_gpu_model(self.ae, gpus=2)
    
    def fitAE(self):
        model = self.ae
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss= custom_loss)
        tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        trained_models_path = 'models/denoising/Denoising-stdev.0.2-'
        model_names = trained_models_path + '.{epoch:02d}-{loss:.4f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_names, monitor='loss', verbose=1, save_best_only=True, save_weights_only = True)
        early_stop = EarlyStopping('loss', patience=20)
        reduce_lr = ReduceLROnPlateau('loss', factor=0.1, patience=int(20 / 4), verbose=1)
        callbacks = [tensor_board,early_stop,reduce_lr,model_checkpoint]
        model.fit_generator(
                self.generator,
                steps_per_epoch=222,
                epochs=100, callbacks = callbacks)

    def create_autoencoder(self):
        input_tensor = Input(shape=(224,224,3))
        #x = GaussianNoise(stddev=0.5)(input_tensor)
        x = ZeroPadding2D((1, 1))(input_tensor)
        x = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(x)
        orig_1 = x
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(x)
        orig_2 = x
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(256, (3, 3), activation='relu', name='conv3_1')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(256, (3, 3), activation='relu', name='conv3_2')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(256, (3, 3), activation='relu', name='conv3_3')(x)
        orig_3 = x
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(512, (3, 3), activation='relu', name='conv4_1')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(512, (3, 3), activation='relu', name='conv4_2')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(512, (3, 3), activation='relu', name='conv4_3')(x)
        orig_4 = x
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(512, (3, 3), activation='relu', name='conv5_1')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(512, (3, 3), activation='relu', name='conv5_2')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(512, (3, 3), activation='relu', name='conv5_3')(x)
        
        orig_5 = x
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv6', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
        
        
        x = BatchNormalization()(x)
        x = UpSampling2D(size=(2, 2))(x)
        #x = Unpooling(orig_5)(x)
        
        #Decoder

        x = Conv2D(512, (5, 5), activation='relu', padding='same', name='deconv5', kernel_initializer='he_normal',
                bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D(size=(2, 2))(x)
        # x = Unpooling(orig_4)(x)

        x = Conv2D(256, (5, 5), activation='relu', padding='same', name='deconv4', kernel_initializer='he_normal',
                bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D(size=(2, 2))(x)
        # x = Unpooling(orig_3)(x)

        x = Conv2D(128, (5, 5), activation='relu', padding='same', name='deconv3', kernel_initializer='he_normal',
                bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D(size=(2, 2))(x)
        # x = Unpooling(orig_2)(x)

        x = Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv2', kernel_initializer='he_normal',
                bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D(size=(2, 2))(x)
        # x = Unpooling(orig_1)(x)

        x = Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1', kernel_initializer='he_normal',
                bias_initializer='zeros')(x)
        x = BatchNormalization()(x)

        x = Conv2D(3, (5, 5), activation='sigmoid', padding='same', name='pred', kernel_initializer='he_normal',
                bias_initializer='zeros')(x)

        model = Model(inputs=input_tensor, outputs=x)
        return model
    

ocnn = OCNN()

#ocnn.ae.load_weights('models/model.93-0.1170.hdf5')
#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#ocnn.ae.compile(optimizer=sgd, loss= custom_loss)

        



    