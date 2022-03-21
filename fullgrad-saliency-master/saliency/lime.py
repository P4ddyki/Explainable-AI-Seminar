#
# Added manually by me to include Lime 
#

"""  

    
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose
import numpy as np
import keras as keras
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

from saliency.tensor_extractor import FullGradExtractor
from keras.applications.vgg16 import (decode_predictions)

class Lime():
    """
    Compute simple FullGrad saliency map 
    """

    def __init__(self, model, im_size = (3,224,224) ):
        self.model = model
        self.model_ext = FullGradExtractor(model, im_size)

    def saliency(self, image, target_class=None):
        explainer = lime_image.LimeImageExplainer()
        preds = self.model.eval(image)

        for i in decode_predictions(preds)[0]:
            print(i)
            explanation = explainer.explain_instance(x.astype('double'), 
                                                    model.predict, 
                                                    top_labels=5, 
                                                    hide_color=0, 
                                                    num_samples=1000)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)

        return deprocess_image(mark_boundaries(temp / 2 + 0.5, mask))

    def deprocess_image(x):
        if np.ndim(x) > 3:
            x = np.squeeze(x)
            # normalize tensor: center on 0., ensure std is 0.1
            x -= x.mean()
            x /= (x.std() + 1e-5)
            x *= 0.1

            # clip to [0, 1]
            x += 0.5
            x = np.clip(x, 0, 1)

            # convert to RGB array
            x *= 255
            if keras.backend.image_data_format() == 'th':
                x = x.transpose((1, 2, 0))
            x = np.clip(x, 0, 255).astype('uint8')
            return x
