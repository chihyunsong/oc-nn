import sys,os
import numpy as np
PROJECT_DIR = '/tf/Hutomize'
sys.path.append(PROJECT_DIR)
from KerasDataset import SurgicalDataset
import tensorflow as tf
from keras.layers import GaussianNoise, Input, Conv2D, UpSampling2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, Dense, Flatten
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
from custom_layers.unpooling_layer import Unpooling
from utils import custom_loss
from keras.optimizers import SGD
from keras.utils.training_utils import multi_gpu_model
import matplotlib.pyplot as plt
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

center = None
R_updated = None
rvalue = 0.1
nu = 0.01
floatX = np.float32
rep_dim = 1568

class Adjust_svdd_Radius(Callback):
   def __init__(self, model, cvar, radius, xtrain, rep_dim):
        self.radius = radius
        self.xtrain_radius = xtrain
        self.cvar = cvar
        self.y_reps = np.zeros((len(xtrain), rep_dim))
        self.model = model
        self.rep_dim = rep_dim
        self.rVar = radius

   def on_epoch_end(self, batch, logs={}):
        reps = self.model.predict(self.xtrain_radius) 
        self.reps = reps
        center = self.cvar
        
        dist = np.sum((reps - self.cvar)**2, axis=1)
        scores = dist
        val = np.sort(scores)
        R_new = np.percentile(val, nu * 100)
        self.rVar = R_new
        

class OCNN(object):

    channels = 3
    image_height = 228
    image_width = 228

    def __init__(self):
        self.data = SurgicalDataset('data', 'OCNN')
        self.generator = self.data.getIterator()
        self.ae = self.create_autoencoder()
        self.ae = multi_gpu_model(self.ae, gpus=2)
        #self.remove_decoder_layers()
        #self.ae = multi_gpu_model(self.ae, gpus=2)


    


    def custom_ocnn_hyperplane_loss(self):
            r = rvalue
            center = self.cvar
            
            def custom_hinge(y_true, y_pred):
                term3 =   K.square(r) + K.sum( K.maximum(0.0,    K.square(y_pred -center) - K.square(r)  ) , axis=1 )
                term3 = 1 / nu * K.mean(term3)

                return term3
            return custom_hinge
            
    def remove_decoder_layers(self):
        model = self.ae
        print(model)
        old_model = self.ae.layers[-2]
        for i in range(0, 16):
            old_model.layers.pop()

        return old_model
    def create_OCNN(self):
        encoder = self.remove_decoder_layers()
        ocnn = Sequential()
        for layer in encoder.layers:
            layer.trainable = False
            ocnn.add(layer)

        ocnn.add(Flatten())
        ocnn.add(Dense(2588, activation='linear'))
        ocnn.add(Dense(1568, activation='linear'))
        ocnn.summary()
        self.h_size = 1568
        return ocnn

    def fitOCNN(self):
        self.ae.load_weights('models/denoising/Denoising-stdev.0.2-2.21-0.0560.hdf5')
       
        self.ocnn = self.create_OCNN()
        self.ocnn = multi_gpu_model(self.ocnn, gpus=2)

        self.train_to_adjust_R = self.data.getTest()

        data = next(self.train_to_adjust_R)[0]
        reps = self.ocnn.predict(data)

        c = np.mean(reps, axis = 0)
        # print(reps)
        eps= 0.1
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.cvar = c
        self.Rvar = 0

        epochs = 200
        lr_power = 0.9
        def lr_scheduler(epoch, mode='adam'):
            # '''if lr_dict.has_key(epoch):
            #     lr = lr_dict[epoch]
            #     print 'lr: %f' % lr'''
            lr = 1e-4
            if mode is 'power_decay':
                # original lr scheduler
                lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
            if mode is 'exp_decay':
                # exponential decay
                lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)
            # adam default lr
            if mode is 'adam' and epoch > 50:
                lr = 1e-5
                if(epoch== 51):
                    print('lr: rate adjusted for fine tuning %f' % lr)

            if mode is 'progressive_drops':
                # drops as progression proceeds, good for sgd
                if epoch > 0.9 * epochs:
                    lr = 0.0001
                elif epoch > 0.75 * epochs:
                    lr = 0.001
                elif epoch > 0.5 * epochs:
                    lr = 0.01
                else:
                    lr = 0.1

            # print('lr: %f' % lr)
            return lr

        scheduler = LearningRateScheduler(lr_scheduler)

        trained_models_path = 'models/OCNN_DENOISING/ocnn'

        model_names = trained_models_path + '.{epoch:02d}-{loss:.4f}.hdf5'
       
        model_checkpoint = ModelCheckpoint(model_names, monitor='loss', verbose=1, save_best_only=True, save_weights_only = True)
        out_batch = Adjust_svdd_Radius(self.ocnn, self.cvar, self.Rvar, data, self.h_size)
        callbacks = [out_batch, scheduler, model_checkpoint]

        opt = Adam(lr=1e-4)
            
        print("[INFO:] Hyperplane Loss function.....")
        self.ocnn.compile(loss=self.custom_ocnn_hyperplane_loss(),
                          optimizer=opt)
        self.ocnn.fit_generator(
                self.generator,
                steps_per_epoch=111,
                epochs=250, callbacks = callbacks)
        self.Rvar = out_batch.radius
        print("[INFO:] \n Model compiled and fit Initial Radius Value...", self.Rvar)
        

    def checkAE(self):
        self.ae.load_weights('models/model.93-0.1170.hdf5')

        test_data = next(self.generator)
        test_result= self.ae.predict(test_data[0])
        print(len(test_result))
        print(len(test_data[0]))
        print(len(test_data[1]))
        for i in range(60):
           plt.imsave('encoder_result/normal/training_decodedImage_'+str(i)+'.png', test_result[i])
           plt.imsave('encoder_result/normal/training_originalImage_'+str(i)+'.png', test_data[1][i])
    def fitAE(self):
        model = self.ae
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss= custom_loss)
        tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        trained_models_path = 'models/denoising/Denoising-stdev.0.2-2'
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
        # x = Unpooling(orig_5)(x)
        
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

        



    