import sys,os
import numpy as np
PROJECT_DIR = '/tf/Hutomize'
sys.path.append(PROJECT_DIR)
from KerasDataset import SurgicalDataset
import tensorflow as tf
from keras.layers import GaussianNoise, Input, Conv2D, UpSampling2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, Dense, Flatten
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
from custom_layers.unpooling_layer import Unpooling
from utils import custom_loss
from keras.optimizers import SGD
from keras.utils.training_utils import multi_gpu_model
import matplotlib.pyplot as plt


center = None
R_updated = None
rvalue = 0.01
nu = 0.01

class Adjust_svdd_Radius(Callback):
   def __init__(self, model, cvar, radius, xtrain, rep_dim):
        self.radius = radius
        self.inputs = inputs
        self.cvar = cvar
        self.y_reps = np.zeros((len(X_trains), rep_dim))
        self.model = model
        self.rep_dim = rep_dim

   def on_epoch_end(self, batch, logs={}):
        reps = self.model.predict(self.inputs[0]) 
        self.reps = reps 
        center = self.cvar


class OCNN(object):

    channels = 3
    image_height = 228
    image_width = 228

    def __init__(self):
        self.data = SurgicalDataset('data')
        self.generator = self.data.getIterator()
        self.ae = self.create_autoencoder()
        #self.ae = multi_gpu_model(self.ae, gpus=2)
        self.remove_decoder_layers()
        self.ocnn = self.create_OCNN()

        #self.ae = multi_gpu_model(self.ae, gpus=2)

        self.ae.summary()
    def create_OCNN(self):
        ocnn = Sequential()
        
        for layer in self.ae:
            ocnn.add(layer)
        for layer in ocnn:
                layer.trainable = False
        print("Set The Layers to be non trainable")

        print(self.ocnn.summary())
        ocnn.add(Flatten())
        ocnn.add(Dense(3136, activation='linear'))
        return ocnn
    def get_reps(self, inputs):
        self.
            
    def initialize_c_and_R(self, inputs):
        ## Initialize  c and R

        reps = self.get_OC_SVDD_network_reps(inputs)
        self.reps = reps

        print("[INFO:] The shape of the reps obtained are", reps.shape)

        reps = np.reshape(reps, (len(reps), (32)))
        self.reps = reps
        print("[INFO:] The shape of the reps obtained are", reps.shape)

        print("[INFO:] Initializing c and Radius R value...")
        eps = 0.1
        # consider the value all the number of batches (and thereby samples) to initialize from
        c = np.mean(reps, axis=0)

        # If c_i is too close to 0 in dimension i, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.cvar = c  # Initialize the center

        # initialize R at the (1-nu)-th quantile of distances
        print("[INFO:] Center (c)  initialized.", c.shape)
        dist_init = np.sum((reps - c) ** 2, axis=1)
        out_idx = int(np.floor(len(reps) * nu))
        sort_idx = dist_init.argsort()
        self.Rvar = floatX(dist_init[sort_idx][-out_idx])

        print("[INFO:] Center (c)  initialized.", c)
        # print("[INFO:] Shape of Center (c)  initialized.", c.shape)
        # print("[INFO:] Distances (D)  initialized.", dist_init)
        # print("[INFO:] Shape of Distances (D)  initialized.", dist_init.shape)
        # print("[INFO:] out_idx (D)  out_idx.", out_idx)
        # print("[INFO:] sort_idx (D)  sort_idx.", sort_idx)
        print("[INFO:] Radius (R)  initialized.", np.float32(dist_init[sort_idx][-out_idx]))

        return

    def remove_decoder_layers(self):
        model = self.ae
        for i in range(0, 21):
            model.layers.pop()
        print (model.summary())

        return model

    def fitOCNN(self):
        self.ae.load_weights('models/Unpooling_Denoising/Unpooling-stdev.0.2..90-0.0391.hdf5')
        self.remove_decoder_layers()

        pass

    def checkAE(self):
        self.ae.load_weights('models/model.93-0.1170.hdf5')

        test_data = next(self.generator)
        test_result= self.ae.predict(test_data[0])
     
        # for i in range(60):
        #    plt.imsave('encoder_result/normal/training_decodedImage_'+str(i)+'.png', test_result[i])
        #    plt.imsave('encoder_result/normal/training_originalImage_'+str(i)+'.png', test_data[1][i])
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

        x = Conv2D(3, (5, 5), activation='sigmoid', padding='same', name='y_pred', kernel_initializer='he_normal',
                bias_initializer='zeros')(x)

        model = Model(inputs=input_tensor, outputs=x)
        print (model.summary())
        return model
        
    def custom_ocnn_loss(self):
        center= self.cvar

        # w = self.oc_nn_model.layers[-2].get_weights()[0]
        # V = self.oc_nn_model.layers[-1].get_weights()[0]
        # print("Shape of w",w.shape)
        # print("Shape of V",V.shape)
        

        def custom_hinge(y_true, y_pred):
            # term1 = 0.5 * tf.reduce_sum(w ** 2)
            # term2 = 0.5 * tf.reduce_sum(V ** 2)

            term3 =   K.square(r) + K.sum( K.maximum(0.0,    K.square(y_pred -center) - K.square(r)  ) , axis=1 )
            # term3 = K.square(r) + K.sum(K.maximum(0.0, K.square(r) - K.square(y_pred - center)), axis=1)
            term3 = 1 / nu * K.mean(term3)

            loss = term3

            return (loss)

        return custom_hinge
           
           

ocnn = OCNN()

#ocnn.ae.load_weights('models/model.93-0.1170.hdf5')
#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#ocnn.ae.compile(optimizer=sgd, loss= custom_loss)

        



