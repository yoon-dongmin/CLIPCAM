class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layer, reshape_transform):
        self.model = model                      # 딥러닝 모델을 저장합니다.
        self.gradients = []                     # 그래디언트를 저장하기 위한 빈 리스트를 생성합니다.
        self.activations = []                   # 활성화를 저장하기 위한 빈 리스트를 생성합니다.
        self.reshape_transform = reshape_transform # 재구성 변환 함수를 저장합니다.

        # PyTorch의 hooking 기능을 사용하여 지정한 계층의 출력값과 그래디언트를 저장합니다.
        target_layer.register_forward_hook(self.save_activation)  # forward pass 동안 해당 계층의 출력값을 저장합니다.
        target_layer.register_backward_hook(self.save_gradient)  # backward pass 동안 해당 계층의 그래디언트를 저장합니다.

    def save_activation(self, module, input, output):
        activation = output # forward pass의 결과인 활성화를 저장합니다.
        # 만약 재구성 변환 함수가 주어져 있다면, 이를 활성화에 적용합니다.
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu()) # 활성화를 리스트에 추가합니다.

    def save_gradient(self, module, grad_input, grad_output):
        # backward pass의 결과인 그래디언트를 저장합니다.
        grad = grad_output[0]
        # 만약 재구성 변환 함수가 주어져 있다면, 이를 그래디언트에 적용합니다.
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu()] + self.gradients # 그래디언트를 리스트에 추가합니다. 그래디언트는 역순으로 계산됩니다.

    def __call__(self, x, x_text):
        self.gradients = []     # 그래디언트 리스트를 초기화합니다.
        self.activations = []   # 활성화 리스트를 초기화합니다.   
        return self.model.forward_mean(x, x_text)  # 모델의 forward pass를 수행하고 결과를 반환합니다.


class ActivationsAndGradients_original:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform

        target_layer.register_forward_hook(self.save_activation)

        #Backward compitability with older pytorch versions:
        if hasattr(target_layer, 'register_full_backward_hook'):
            target_layer.register_full_backward_hook(self.save_gradient)
        else:
            target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)