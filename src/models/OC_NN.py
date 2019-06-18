# import the necessary packages
import numpy as np
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import tensorflow as tf
sess = tf.Session()
import keras


from keras import backend as K
K.set_session(sess)

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.utils import to_categorical
# set the matplotlib backend so figures can be saved in the background
from keras.callbacks import LambdaCallback
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,Adagrad
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import cv2 
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects
from src.data.main import load_dataset
from keras.layers import Activation, LeakyReLU, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, \
    BatchNormalization, regularizers
from src.config import Configuration as Cfg

class OC_NN:

    ## Initialise static variables
    INPUT_DIM = 0
    HIDDEN_SIZE = 0
    DATASET = ""

    def __init__(self, dataset, inputdim,hiddenLayerSize,img_hgt,img_wdt,modelSavePath,reportSavePath,preTrainedWtPath, seed=42, mnist_normal=0, stringParam="defaultValue",
                 otherParam=None):
        """
        Called when initializing the classifier
        """
        OC_NN.DATASET = dataset
        OC_NN.INPUT_DIM = inputdim
        OC_NN.HIDDEN_SIZE = hiddenLayerSize
        self.stringParam = stringParam

        # THIS IS WRONG! Parameters should have same name as attributes
        self.differentParam = otherParam

        self.directory = modelSavePath
        self.results = reportSavePath
        self.pretrainedWts = preTrainedWtPath
        self.model = ""
        self.IMG_HGT = img_hgt
        self.IMG_WDT = img_wdt
        self.h_size= OC_NN.HIDDEN_SIZE
        global model
        self.r=1.0
        self.kvar=0.0
        self.model = None

        Cfg.seed = seed
        Cfg.mnist_normal = mnist_normal

        load_dataset(self, dataset.lower(), False)

    def load_data(self, data_loader=None, pretrain=False):
        self.data = data_loader()
        print(self.data)
        return

    @staticmethod
    def image_to_feature_vector(image, IMG_HGT,IMG_WDT):
        # resize the image to a fixed size, then flatten the image into
        # a list of raw pixel intensities
        return np.reshape(image,(len(image),IMG_HGT*IMG_WDT))

    @staticmethod
    def custom_ocnn_loss(self,nu, w, V):

        def custom_hinge(y_true, y_pred):

            term1 = 0.5 * tf.reduce_sum(w[0] ** 2)
            term2 = 0.5 * tf.reduce_sum(V[0] ** 2)
            term3 = 1 / nu * K.mean(K.maximum(0.0, self.r - tf.reduce_max(y_pred, axis=1)), axis=-1)
            term4 = -1*self.r
            # yhat assigned to r
            self.r = tf.reduce_max(y_pred, axis=1)
            # r = nuth quantile
            self.r = tf.contrib.distributions.percentile(self.r, q=100 * nu)
            rval = tf.reduce_max(y_pred, axis=1)
            rval = tf.Print(rval, [tf.shape(rval)])



            return (term1 + term2 + term3 + term4)

        return custom_hinge

    @staticmethod
    def buildUsingEncoder(width, height, classes, encoder):
        inp = encoder.input
        out = encoder.layers[-1].output
        from keras.models import Model
        encoder_model = Model(inp, out)  # create a new model which doesn't have the last two layers in VGG16

        model = Sequential()
        for layer in encoder_model.layers:
            model.add(layer)

        h_size = OC_NN.HIDDEN_SIZE

        def custom_activation(x):
            return (1 / np.sqrt(h_size)) * tf.cos(x / 0.02)

        get_custom_objects().update({
            'custom_activation':
                Activation(custom_activation)
        })

        # main thread

        ## Define Dense layer from input to hidden
        model.add(Flatten())
        input_hidden = Dense(h_size, kernel_initializer="glorot_normal", name="input_hidden")
        model.add(input_hidden)
        model.add(Activation(custom_activation))

        ## Define Dense layer from hidden  to output
        hidden_ouput = Dense(classes, name="hidden_output")
        model.add(hidden_ouput)
        model.add(Activation("linear"))

        ## Obtain the weights and bias of the layers
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        # w = [w.eval(K.get_session) for w in layer_dict['input_hidden'].weights]
        with sess.as_default():
            w = input_hidden.get_weights()[0]
            bias1 = input_hidden.get_weights()[1]
            V = hidden_ouput.get_weights()[0]
            bias2 = hidden_ouput.get_weights()[1]

        ## Load the pretrained model
        # model = load_model("/Users/raghav/envPython3/experiments/one_class_neural_networks/models/FF_NN/" + "FF_NN_best.h5")
        # return the constructed network architecture
        return [model, w, V, bias1, bias2]

    @staticmethod
    def build(width, height, classes):

        h_size = OC_NN.HIDDEN_SIZE

        def custom_activation(x):

            return (1 / np.sqrt(h_size)) * tf.cos(x / 0.02)

        get_custom_objects().update({
            'custom_activation':
                Activation(custom_activation)
        })

        # main thread


        model = Sequential()
        ## Define Dense layer from input to hidden
        input_hidden= Dense(h_size, input_dim= OC_NN.INPUT_DIM, kernel_initializer="glorot_normal",name="input_hidden")
        model.add(input_hidden)
        model.add(Activation(custom_activation))

        ## Define Dense layer from hidden  to output
        hidden_ouput = Dense(classes,name="hidden_output")
        model.add(hidden_ouput)
        model.add(Activation("linear"))

        ## Obtain the weights and bias of the layers
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        # w = [w.eval(K.get_session) for w in layer_dict['input_hidden'].weights]
        with sess.as_default():
            w = input_hidden.get_weights()[0]
            bias1 = input_hidden.get_weights()[1]
            V = hidden_ouput.get_weights()[0]
            bias2 = hidden_ouput.get_weights()[1]

        ## Load the pretrained model
        # model = load_model("/Users/raghav/envPython3/experiments/one_class_neural_networks/models/FF_NN/" + "FF_NN_best.h5")
        # return the constructed network architecture
        return [model,w,V,bias1,bias2]
    def get_oneClass_trainData(self):
        if (OC_NN.DATASET == "mnist"):
            X_train = self.data._X_train
            y_train = self.data._y_train

            X_test = self.data._X_test
            y_test = self.data._y_test

            ## Combine the positive data
            trainXPos = X_train[np.where(y_train == 0)]
            trainYPos = np.zeros(len(trainXPos))

            self.testXPos = X_test[np.where(y_test == 0)]
            testYPos = np.zeros(len(self.testXPos))

            # Combine the negative data
            trainXNeg = X_train[np.where(y_train == 1)]
            trainYNeg = np.ones(len(trainXNeg))

            self.testXNeg = X_test[np.where(y_test == 1)]
            testYNeg = np.ones(len(self.testXNeg))

            X_trainPOS = np.concatenate((trainXPos, self.testXPos))
            y_trainPOS = np.concatenate((trainYPos, testYPos))

            X_trainNEG = np.concatenate((trainXNeg, self.testXNeg))
            y_trainNEG = np.concatenate((trainYNeg, testYNeg))

            # Just 0.01 points are the number of anomalies.
            num_of_anomalies = int(0.01 * len(X_trainPOS))
            X_trainNEG = X_trainNEG[0:num_of_anomalies]
            y_trainNEG = y_trainNEG[0:num_of_anomalies]

            X_train = np.concatenate((X_trainPOS, X_trainNEG))
            y_train = np.concatenate((y_trainPOS, y_trainNEG))

            print("[INFO: ] Shape of One Class Input Data used in training", X_train.shape)
            print("[INFO: ] Shape of (Positive) One Class Input Data used in training", X_trainPOS.shape)
            print("[INFO: ] Shape of (Negative) One Class Input Data used in training", X_trainNEG.shape)


            return X_train

        return
    def fitUsingEncoder(self):
        EPOCHS = 150
        INIT_LR = 1e-8
        BS = 100
        print("[INFO] compiling model...")
        trainX = self.get_oneClass_trainData()

        trainY = np.ones(len(trainX))
        trainY = to_categorical(trainY,
                                num_classes=2)  ## trainY is not used while training its just used since defining keras custom loss function required it
        [cae, encoder] = self.pretrain_cae(solver="adam", lr=1.0, n_epochs=150)
        inp = encoder.input
        out = encoder.layers[-1].output
        from keras.models import Model
        encoderModel = Model(inp, out)  # create a new model which doesn't have the last two layers in VGG16

        [self.model, self.w, self.V, bias1, bias2] = OC_NN.buildUsingEncoder(width=self.IMG_HGT, height=self.IMG_WDT, classes=2, encoder=encoderModel)
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        ## Obtain weights of layer
        nu = 0.01
        print("[INFO] ", self.w[0].shape, "input  --> hidden layer weights shape ...")
        print("[INFO] ", self.V[0].shape, "hidden --> output layer weights shape ...")
        self.model.compile(loss=OC_NN.custom_ocnn_loss(self, nu, self.w, self.V), optimizer=opt, metrics=None)
        output_layers = ['hidden_output']
        self.model.metrics_tensors += [layer.output for layer in self.model.layers if layer.name in output_layers]
        # train the network
        print("[INFO] training network...")

        def printEvaluation(e, logs):
            print("evaluation for epoch: " + str(e))
            print("output:", K.print_tensor(self.model.metrics_tensors[0], message="tensors is: "))

        callback = LambdaCallback(on_epoch_end=printEvaluation)

        tbCallBack = keras.callbacks.TensorBoard(log_dir='../graph', histogram_freq=0, write_graph=True,
                                                 write_images=True)
        callbacks = [callback, tbCallBack]

        ## Initialize the network with pretrained weights
        # model.load_weights(self.pretrainedWts + "FF_NN_weightsfile.h5")


        H = self.model.fit(trainX, trainY, shuffle=False, batch_size=BS, epochs=EPOCHS, validation_split=0.0, verbose=1,
                          callbacks=callbacks)


        # print("[INFO] ",type(w) ,w.shape,"type of w...")
        # print("[INFO] ", type(V),V.shape, "type of V...")
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        # plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
        # plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
        plt.title("OC_NN Training Loss and Accuracy on 1's / 7's")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Vs Epochs")
        plt.legend(loc="upper right")
        plt.savefig(self.results + "trainValLoss-usingEncoder.png")

    def fit(self):
        # initialize the model
        EPOCHS = 1000
        INIT_LR = 1e-8
        BS = 100
        print("[INFO] compiling model...")
        trainX = self.get_oneClass_trainData()
        trainX= OC_NN.image_to_feature_vector(trainX, self.IMG_HGT, self.IMG_WDT)
        trainY = np.ones(len(trainX))
        trainY = to_categorical(trainY, num_classes=2) ## trainY is not used while training its just used since defining keras custom loss function required it

        [model, w, V, bias1, bias2] = OC_NN.build(width=self.IMG_HGT, height=self.IMG_WDT, classes = 2)
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        ## Obtain weights of layer
        nu=0.01
        print("[INFO] ",w[0].shape, "input  --> hidden layer weights shape ...")
        print("[INFO] ",V[0].shape, "hidden --> output layer weights shape ...")
        model.compile(loss=OC_NN.custom_ocnn_loss(self,nu, w, V), optimizer=opt,metrics=None)
        output_layers = ['hidden_output']
        model.metrics_tensors += [layer.output for layer in model.layers if layer.name in output_layers]
        # train the network
        print("[INFO] training network...")

        def printEvaluation(e, logs):
            print("evaluation for epoch: " + str(e) )
            print("output:",K.print_tensor(model.metrics_tensors[0], message="tensors is: "))


        callback = LambdaCallback(on_epoch_end=printEvaluation)

        tbCallBack = keras.callbacks.TensorBoard(log_dir='../graph', histogram_freq=0, write_graph=True,
                                                 write_images=True)
        callbacks = [callback,tbCallBack]


        ## Initialize the network with pretrained weights
        #model.load_weights(self.pretrainedWts + "FF_NN_weightsfile.h5")


        if(OC_NN.DATASET=="USPS"): # validation set is set to 0.0 in case USPS due to lack of data
            H = model.fit(trainX, trainY, shuffle=False,batch_size=BS,epochs=EPOCHS,validation_split=0.0, verbose=1,callbacks=callbacks)
        else:
            H = model.fit(trainX, trainY, shuffle=False, batch_size=BS, epochs=EPOCHS, validation_split=0.0, verbose=1,
                          callbacks=callbacks)

        # save the model to disk
        print("[INFO] serializing network and saving trained weights...")
        print("[INFO] Saving model layer weights..." )
        model.save(self.directory+"OC_NN.h5")
        with sess.as_default():
            w = model.layers[0].get_weights()[0]
            V = model.layers[2].get_weights()[0]
            np.save(self.directory+"w", w)
            np.save(self.directory +"V", V)

        # print("[INFO] ",type(w) ,w.shape,"type of w...")
        # print("[INFO] ", type(V),V.shape, "type of V...")
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        # plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
        # plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
        plt.title("OC_NN Training Loss and Accuracy on 1's / 7's")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Vs Epochs")
        plt.legend(loc="upper right")
        plt.savefig(self.results+"trainValLoss.png")


    def compile_autoencoder(self):
        chanDim = -1  # since depth is appearing the end

        input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

        x = Conv2D(16, (3, 3), use_bias=False, padding='same')(input_img)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
        x = Conv2D(4, (3, 3), padding='same', use_bias=False)(encoded)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(16, (3, 3), use_bias=False)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)

        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', use_bias=False)(x)
        from keras.models import Model
        # this model maps an input to its encoded representation
        encoder = Model(input_img, encoded)

        autoencoder = Model(input_img, decoded)
        # Compile the autoencoder with the mean squared error
        autoencoder.compile(loss='mean_squared_error', optimizer='adam')
        print("Autoencoder Architecture", autoencoder.summary())

        return [autoencoder, encoder]
    def pretrain_cae(self,solver, lr,  n_epochs):
        [cae, encoder] = self.compile_autoencoder()
        X = self.get_oneClass_trainData()

        print("[INFO:] The shape of X used to train CAE", X.shape)

        cae.fit(X, X,
                batch_size=200,
                epochs=n_epochs,
                verbose=0)
        return [cae, encoder]

    def scoreUsingEncoder(self):
        testPosX = self.testXPos
        testNegX = self.testXNeg
        # load the trained convolutional neural network
        print("[INFO] loading network...")
        nu = 0.01






        ## Initialize the network with pretrained weights

        ## y_true
        y_true_pos = np.ones(testPosX.shape[0])
        y_true_neg = np.zeros(testNegX.shape[0])
        y_true_pos = to_categorical(y_true_pos, num_classes=2)
        y_true_neg = to_categorical(y_true_neg, num_classes=2)
        y_true = np.concatenate((y_true_pos, y_true_neg))

        x_test = np.concatenate((testPosX, testNegX), axis=0)

        IMG_HGT = self.IMG_HGT
        IMG_WDT = self.IMG_WDT

        print(y_true.shape[0], 'Actual test samples')
        print(x_test.shape[0], 'X INPUT')

        from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
        y_pred_keras = self.model.predict_proba(x_test)

        y_pred = np.argmax(y_pred_keras, axis=1)
        y_true = np.argmax(y_true, axis=1)

        # print "y_pred.shape",y_pred.shape
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ####
        print ("=" * 35)
        print ("auccary_score:", accuracy)
        print ("roc_auc_score:", auc)
        start = len(x_test) - 100  # Print the last 100 labels among which last 50 are known anomalies
        end = len(x_test)
        print("y_true", y_true[start:end])
        print("y_pred", y_pred[start:end])
        print ("=" * 35)

        return auc
    def score(self):
        testPosX = self.testXPos
        testNegX = self.testXNeg
        # load the trained convolutional neural network
        print("[INFO] loading network...")
        nu=0.01

        w =  np.load(self.directory + "w.npy")
        V =  np.load(self.directory + "V.npy")

        model = load_model(self.directory+"OC_NN.h5",custom_objects={'custom_hinge': OC_NN.custom_ocnn_loss(self,nu,w,V)})
        ## Initialize the network with pretrained weights

        ## y_true
        y_true_pos = np.ones(testPosX.shape[0])
        y_true_neg = np.zeros(testNegX.shape[0])
        y_true_pos = to_categorical(y_true_pos, num_classes=2)
        y_true_neg = to_categorical(y_true_neg, num_classes=2)
        y_true = np.concatenate((y_true_pos, y_true_neg))





        x_test =  np.concatenate((testPosX, testNegX), axis=0)

        IMG_HGT = self.IMG_HGT
        IMG_WDT=self.IMG_WDT
        x_test = OC_NN.image_to_feature_vector(x_test, IMG_HGT, IMG_WDT)


        print(y_true.shape[0], 'Actual test samples')



        from sklearn.metrics import roc_curve,accuracy_score,roc_auc_score
        y_pred_keras = model.predict_proba(x_test)


        y_pred = np.argmax(y_pred_keras, axis=1)
        y_true = np.argmax(y_true, axis=1)

        # print "y_pred.shape",y_pred.shape
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ####
        print ("=" * 35)
        print ("auccary_score:", accuracy)
        print ("roc_auc_score:", auc)
        start = len(x_test) - 100  # Print the last 100 labels among which last 50 are known anomalies
        end = len(x_test)
        print("y_true", y_true[start:end])
        print("y_pred", y_pred[start:end])
        print ("=" * 35)

        return auc




