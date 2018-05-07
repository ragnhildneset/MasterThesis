'''
    Generalized version of VisualBackprop implementation by:
    https://github.com/experiencor/deep-viz-keras
    Which only supported vgg16
'''

import numpy as np
import keras.backend as K
from keras.layers import Input, Conv2DTranspose
from keras.models import Model
from keras.initializers import Ones, Zeros


class VisualBackprop():
    """
        Computes a saliency mask with VisualBackprop
        https://arxiv.org/abs/1611.05418
    """

    def __init__(self, model, output_index=0):
        inps = [model.input, K.learning_phase()]         # input placeholder
        outs = [layer.output for layer in model.layers]  # all layer outputs
        self.forward_pass = K.function(inps, outs)       # evaluation function

        self.model = model

    def get_mask(self, input_image):
        """
            Returns a VisualBackprop mask for an image.
        """
        x_value = np.expand_dims(input_image, axis=0)
        visual_bpr = None
        layer_outs = self.forward_pass([x_value, 0])

        conv_layers = []
        for layer, layer_out in zip(self.model.layers, layer_outs):
            if 'Conv2D' in str(type(layer)):
                conv_layers.append((layer, layer_out))

        # Iterate over feature maps upstream
        for i in range(len(conv_layers)-1, -1, -1):
            # Average the feature map
            layer = np.mean(conv_layers[i][1], axis=3, keepdims=True)

            if visual_bpr is not None:
                if visual_bpr.shape != layer.shape:
                    visual_bpr = self._deconv(visual_bpr,
                                              conv_layers[i+1][0])
                    # If upsampling fails
                    if visual_bpr.shape != layer.shape:
                        visual_bpr = self._add_padding_to_match_shape(
                                            visual_bpr,
                                            layer.shape)
                visual_bpr = visual_bpr * layer  # Pointwise product
            else:
                visual_bpr = layer

        # Last upsampling to input image size
        visual_bpr = self._deconv(visual_bpr, self.model.layers[0])
        visual_bpr = self._add_padding_to_match_shape(
                            visual_bpr, x_value[:, :, :, :1].shape)
        return visual_bpr[0]

    def _add_padding_to_match_shape(self, visual_bpr, target_shape):
        """
            Add one pixel of padding in top and left side of mask if the deconv
            operation doesn't upsample to the exact size of upstream layer.
            (If it fails, it is usually off by one pixel less than wanted)
        """
        # Check difference between upsampled feature map shape and wanted
        # shape in every dimension
        diff = [target_shape[i] - visual_bpr.shape[i]
                for i in range(len(visual_bpr.shape))]
        for axis in range(len(diff)):
            if diff[axis] > 0:
                visual_bpr = np.insert(visual_bpr, 0, 0, axis=axis)
        return visual_bpr

    def _deconv(self, feature_map, upstream_feature_map):
        """
            The deconvolution operation to upsample the average
            feature map upstream.
        """
        x = Input(shape=(None, None, 1))
        y = Conv2DTranspose(filters=1,
                            kernel_size=upstream_feature_map.kernel_size,
                            strides=upstream_feature_map.strides,
                            kernel_initializer=Ones(),
                            activation=upstream_feature_map.activation,
                            use_bias=upstream_feature_map.use_bias,
                            bias_initializer=Zeros())(x)

        deconv_model = Model(inputs=[x], outputs=[y])

        inps = [deconv_model.input, K.learning_phase()]   # input placeholder
        outs = [deconv_model.layers[-1].output]           # output placeholder

        deconv_func = K.function(inps, outs)              # evaluation function

        return deconv_func([feature_map, 0])[0]
