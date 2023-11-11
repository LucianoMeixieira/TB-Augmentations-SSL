'''
This code has been adapted from the work at https://github.com/sivaramakrishnan-rajaraman/CXR-bone-suppression.git
'''

from keras.layers import Input, Conv2D, Add, Lambda
from keras.models import Model
import numpy as np

# The ResNet Bone Suppression Model Architecture that loads in the weights
class ResNetBSModel:
    def __init__(self, weights_path, num_filters=64, num_res_blocks=16, res_block_scaling=0.1):
        # create the model architecture
        self.model = self._build_model(num_filters, num_res_blocks, res_block_scaling)
        
        # load in the weights
        self.model.load_weights(weights_path)

    def _build_model(self, num_filters, num_res_blocks, res_block_scaling):
        x_in = Input(shape=(256, 256, 1))
        x = b = Conv2D(num_filters, (3, 3), padding='same')(x_in)
        for i in range(num_res_blocks):
            b = self._res_block(b, num_filters, res_block_scaling)
        b = Conv2D(num_filters, (3, 3), padding='same')(b)
        x = Add()([x, b])
        x = Conv2D(1, (3, 3), padding='same')(x)
        return Model(x_in, x, name="ResNet-BS")
    
    def _res_block(self, x_in, filters, scaling):
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x_in)
        x = Conv2D(filters, (3, 3), padding='same')(x)
        if scaling:
            x = Lambda(lambda t: t * scaling)(x)
        x = Add()([x_in, x])
        return x
        
    # get the bone suppression prediction 
    # this function will work for a single image or a batch of images
    def predict(self, input_data, verbose):
        # If the input data is of shape (256, 256), it's a single image, so we expand dimensions
        if len(input_data.shape) == 2:
            input_data = np.expand_dims(input_data, axis=0)  # Adding batch dimension
            
        # Adding channel dimension, regardless of whether it's a single image or a batch
        input_data = np.expand_dims(input_data, axis=-1)
        
        # Getting the model prediction
        prediction = self.model.predict(input_data, verbose=verbose)
        
        # If it was a single image, we remove the batch dimension, else we keep it
        if len(prediction.shape) == 4 and prediction.shape[0] == 1:  # It was a single image (1, 256, 256, 1) --> batch size = 1 at index 0
            prediction = np.squeeze(prediction, axis=0)  # Remove the batch dimension
        prediction = np.squeeze(prediction, axis=-1)  # Remove the channel dimension
        
        return prediction  # Returning the bone suppressed prediction