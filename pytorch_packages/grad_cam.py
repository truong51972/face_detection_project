import torch
from torchvision import transforms

from PIL import Image

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class Grad_cam:
    def __init__(self, model, model_name):

        self.model = model

        # resnet
        if model_name == "resnet50" or model_name == "resnet18":
            target_layers = [self.model.layer4[-1]]
        elif model_name == "vgg16" or model_name == "vgg19":
            target_layers = [self.model.features[-1]]
        elif model_name == "effi_net_v2_s" or model_name == "effi_net_v2_s":
            target_layers = [self.model.features[-1]]
            
        # self.cam = GradCAM(model=self.model, target_layers=target_layers)
        # self.cam = ScoreCAM(model=self.model, target_layers=target_layers)
        # self.cam = HiResCAM(model=self.model, target_layers=target_layers)
        self.cam = GradCAMPlusPlus(model=self.model, target_layers=target_layers)
        # self.cam = AblationCAM(model=self.model, target_layers=target_layers)

        self.img_transform_for_visualize = transforms.Compose([
            transforms.Resize(size= 224),
            transforms.ToTensor(),
        ])

    # @time_decorator
    def visualize(self, img, predict, threshold= 0.3):
        img_tensor = self.img_transform_for_visualize(img)
        img_tensor_in_batch = img_tensor.unsqueeze(dim= 0)
        
        rgb_img = img_tensor.permute(1, 2, 0).numpy()
        
        targets = [ClassifierOutputTarget(torch.argmax(predict, dim= 1).item())]
        
        grayscale_cam = self.cam(input_tensor=img_tensor_in_batch, targets= targets)[0]
        
        # grayscale_cam[grayscale_cam < threshold] = 0
        
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        visualization = Image.fromarray(visualization)

        return grayscale_cam, visualization
