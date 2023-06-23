import cv2
import numpy as np
import torch
import ttach as tta
from pytorch_grad_cam_modified.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam_modified.utils.svd_on_activations import get_2d_projection


class BaseCAM:
    # BaseCAM 생성자 함수
    def __init__(self, 
                 model,  # CAM을 적용하고 싶은 신경망 모델
                 target_layer,  # CAM을 얻고 싶은 대상 계층
                 gpu_id=0,  # 연산을 수행할 GPU의 번호
                 reshape_transform=None):  # 입력 이미지를 변형하는 함수
                 
        # 모델을 평가 모드로 설정하고, 지정된 gpu에 할당합니다.
        self.model = model.eval().to(gpu_id)  
        self.target_layer = target_layer  # 대상 계층 저장
        self.gpu_id = gpu_id  # GPU id 저장
        self.reshape_transform = reshape_transform  # 재형성 변환 저장
        
        # ActivationsAndGradients 객체 생성
        # 특정 계층에서의 활성화값과 그래디언트를 저장하고 관리합니다.
        self.activations_and_grads = ActivationsAndGradients(self.model, 
            target_layer, reshape_transform)
            
        self.text_tensor = 0  # 사용자의 입력을 저장할 변수 초기화
        self.input_tensor = 0  # 사용자의 입력을 저장할 변수 초기화


    #forward가 2개?

    # def forward(self, input_img):
    #     return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss


    def get_cam_image(self,
                  input_tensor,
                  target_category,
                  activations,
                  grads,
                  eigen_smooth=False):
        # get_cam_weights 함수를 호출하여 Class Activation Map(CAM)을 계산하기 위한 가중치를 얻습니다.
        # 해당 함수는 grad_cam.py에 존재
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        # print(weights.shape)

        # 계산된 가중치를 활성화 값에 곱하여 가중치가 적용된 활성화를 생성합니다.
        # weight의 차원을 2개 추가
        weighted_activations = weights[:, :, None, None] * activations
        # print(weighted_activations.shape) #(1,768,7,7)
        # print(activations.shape)

        
        # eigen_smooth 플래그에 따라 다른 계산 방법을 사용합니다.
        if eigen_smooth:
            # eigen_smooth가 True인 경우, weighted_activations에 2D 투영을 적용하여 CAM을 계산합니다.
            cam = get_2d_projection(weighted_activations)
        else:
            # eigen_smooth가 False인 경우, weighted_activations을 축 1(axis=1)을 따라 합산하여 CAM을 계산합니다.
            # 채널을 기준으로 sum을 함
            cam = weighted_activations.sum(axis=1)
            # print(cam.shape) #(1,7,7)

        # 계산된 CAM 이미지를 반환합니다.
        return cam


    # def get_cam_image(self,
    #                   input_tensor,
    #                   target_category,
    #                   activations,
    #                   grads,
    #                   eigen_smooth=False):
    #     weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
    #     weighted_activations = weights[:, :, None, None] * activations
    #     if eigen_smooth:
    #         cam = get_2d_projection(weighted_activations)
    #     else:
    #         cam = weighted_activations.sum(axis=1)
    #     return cam

    def forward(self, input_tensor, text_tensor, target_category=None, eigen_smooth=False, compute_text=False):
        # 입력 텐서들을 객체 내부에 저장합니다.
        self.text_tensor = text_tensor
        self.input_tensor = input_tensor

        # ActivationsAndGradients 객체를 사용하여 주어진 입력에 대해 모델을 실행하고 출력값을 얻음
        output, _ = self.activations_and_grads(input_tensor, text_tensor)

        # target_category는 모델이 예측해야 하는 목표 카테고리입니다.
        # 만약 target_category가 정수라면, 입력 텐서의 모든 이미지에 대해 동일한 카테고리를 타겟으로 합니다.
        # target_category가 None이라면, 모델의 출력에서 가장 높은 확률을 가진 카테고리를 선택합니다. => 0
        if type(target_category) is int:
            target_category = [target_category] * input_tensor.size(0)
        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)

        # 모델의 그래디언트를 0으로 초기화하고, 역전파를 수행하여 그래디언트를 계산합니다.
        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        # print(output) #260
        # print(target_category) #0
        # print(loss) #260
        loss.backward(retain_graph=True)

        # 그래디언트와 활성화를 가져와 CAM을 생성합니다.
        # 최근 forward pass에서 target layer에서의 활성화 값을 가져와 numpy 배열로 변환합니다.
        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()

        # 마찬가지로, 최근 backward pass에서 target layer에서의 그래디언트를 가져와 numpy 배열로 변환합니다.
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

        # 활성화 값과 그래디언트를 사용하여 CAM 이미지를 생성합니다.
        # text_tensor는 입력 텍스트 텐서, target_category는 CAM을 생성하려는 대상 클래스를 의미하고,
        # eigen_smooth는 CAM을 부드럽게 만드는 옵션입니다.
        cam = self.get_cam_image(text_tensor, target_category, activations, grads, eigen_smooth)

        # print(cam)
        # print(cam.shape) #(1,7,7)
   
        # CAM을 클리핑하여 0 이하의 값들을 0으로 설정합니다.
        cam = np.maximum(cam, 0)
        
        # print(len(cam))
      
        result = [] # 결과를 저장할 리스트 초기화
        for img in cam: # 각각의 Class Activation Map 이미지에 대해
            # 각 CAM 이미지를 float32 타입으로 변환합니다.
            img = np.float32(img)
            
            # compute_text가 False라면, CAM 이미지의 크기를 원본 입력 이미지의 크기와 동일하게 조정합니다.
            if not compute_text:
                # 입력 이미지의 크기에 맞게 CAM 이미지의 크기를 조정합니다.
                img = cv2.resize(img, input_tensor.shape[-2:][::-1]) #너비,높이 가져옴

            # 이미지를 정규화합니다.
            # 이는 이미지의 값 범위를 [0, 1]로 만들기 위한 작업입니다.
            img = img - np.min(img)
            img = img / np.max(img)
            
            # 정규화한 이미지를 결과 리스트에 추가합니다.
            result.append(img)

        # 이 모든 CAM 이미지를 배열로 변환하여 반환합니다.
        result = np.float32(result)
        return result



    # def forward(self, input_tensor, text_tensor, target_category=None, eigen_smooth=False, compute_text=False):

    #     self.text_tensor = text_tensor
    #     self.input_tensor = input_tensor

    #     # if self.cuda:
    #     #     input_tensor = input_tensor.cuda()
    #     #     text_tensor = text_tensor.cuda()

    #     #logit per image반환
    #     output, _ = self.activations_and_grads(input_tensor, text_tensor)

    #     if type(target_category) is int:
    #         target_category = [target_category] * input_tensor.size(0)

    #     if target_category is None:
    #         target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
    #         # print(target_category)
    #     else:
    #         assert(len(target_category) == input_tensor.size(0))

    #     self.model.zero_grad()
    #     loss = self.get_loss(output, target_category)
    #     loss.backward(retain_graph=True)

    #     activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
    #     grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()
    #     # print(len(activations[0]))
    #     # print(len(grads[0]))

    #     cam = self.get_cam_image(text_tensor, target_category, activations, grads, eigen_smooth)

    #     cam = np.maximum(cam, 0)
    #     #print(cam)
    #     result = []
    #     for img in cam:
    #         img = np.float32(img)
    #         if not compute_text:
    #             img = cv2.resize(img, input_tensor.shape[-2:][::-1])
    #         img = img - np.min(img)
    #         img = img / np.max(img)
    #         result.append(img)
    #     result = np.float32(result)
    #     return result

    def forward_augmentation_smoothing(self,
                                       input_tensor,
                                       target_category=None,
                                       eigen_smooth=False):
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                target_category, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor,
                 text_tensor,
                 target_category=None,
                 aug_smooth=False,
                 eigen_smooth=False,
                 compute_text=False):
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor, text_tensor,
                target_category, eigen_smooth, compute_text)

        return self.forward(input_tensor, text_tensor,
            target_category, eigen_smooth, compute_text)
