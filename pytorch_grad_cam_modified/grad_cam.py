import cv2
import numpy as np
import torch
from pytorch_grad_cam_modified.base_cam import BaseCAM

###BaseCAM에서 상속 받아 사용###
class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, gpu_id=0, 
        reshape_transform=None):
        super(GradCAM, self).__init__(model, target_layer, gpu_id, reshape_transform) #GradCAM 호출

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        # (Batch size, Channel, Height, Width)의 4차원 형태
        #  Height와 Width에 대한 평균
        return np.mean(grads, axis=(2, 3)) #2,3축을 따라 평균을 구함
