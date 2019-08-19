import sys, os
import os.path as path
import numpy as np
import tensorflow as tf
PROJECT_DIR = "/tf/Hutomize/"
WIDTH = 1280
HEIGHT = 1024
CHANNELS = 3
sys.path.append(PROJECT_DIR)
import numpy as np
import PIL.Image as pilimg
import matplotlib

class SurgicalDataset(object):
    def __init__(self, data_path = ''):
        self.skipFrame = 1
        self.dataPath = data_path
        self.dataset = None
        self.totalVideo = []
        self.dataVideo = []
        self.label = []
        ######VIDEO 1   
        i = 0
        file_name = PROJECT_DIR + self.dataPath + '/video1/frame'+ str(i)+'.jpg'

        while path.isfile(file_name):
            i = i + self.skipFrame
            file_name = PROJECT_DIR + self.dataPath + '/video1/frame'+ str(i)+'.jpg'
            self.totalVideo.append(file_name)
        ## video 2 

        i = 0
        file_name = PROJECT_DIR + self.dataPath + '/video2/frame'+ str(i)+'.jpg'

        while path.isfile(file_name):
            i = i + self.skipFrame
            file_name = PROJECT_DIR + self.dataPath + '/video2/frame'+ str(i)+'.jpg'
            self.totalVideo.append(file_name)

    def prepareData(self, skipFrame = 30, batch_size=50):
        '''
            Prepare Data for One Class Neural Network
        '''

        def load_image(path):
            image_string = tf.read_file(path)

            # Don't use tf.image.decode_image, or the output shape will be undefined
            image = tf.image.decode_jpeg(image_string, channels=3)

            # This will convert to float values in [0, 1]
            image = tf.image.convert_image_dtype(image, tf.float32)

            image = tf.image.resize_images(image, [270 , 256])
            cropped = tf.random_crop(images, [224,224,3])  
           
            return cropped

        self.dataVideo = self.totalVideo[::skipFrame]
        self.labels = [0] * len(self.dataVideo)
        self.dataset = tf.data.Dataset.from_tensor_slices((self.dataVideo, self.labels))

        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels =3)
            image = tf.image.resize(image_decoded,(270,256))
            cropped = tf.random_crop(image, [224,224,3]) 
            cropped = cropped/ 255.0
            labels = [0]

            return cropped, labels
        self.dataset = self.dataset.map(_parse_function)
        self.dataset = self.dataset.batch(batch_size).prefetch(batch_size*4)
        
        self.iterator = self.dataset.make_one_shot_iterator()
        

        return self.iterator 
        

        

 