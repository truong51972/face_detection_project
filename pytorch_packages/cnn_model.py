import torch
from torchvision import transforms

from pytorch_packages.model_builder import load_model
from PIL import Image

import asyncio

class Cnn_model:
    def __init__(self, path_to_model: str, device= 'cpu'):
        self.device = device

        self.model, info_data = load_model(pretrain_model_path= path_to_model)
        self.class_names = info_data['class_names']
        self.model_name = info_data['model']

        self.img_transform_for_predict = transforms.Compose([
            transforms.Resize(size= 224),
            transforms.ToTensor(),
        ])

        self.class_name_to_idx = dict(zip(self.class_names, range(len(self.class_names))))
        
    def get_idx(self, class_name):
        try:
            return self.class_name_to_idx[class_name]
        except:
            return None
            
    def get_model(self):
        return self.model

    # @time_decorator
    def _predict(self, img: Image):
        img_tensor = self.img_transform_for_predict(img)
        img_tensor_in_batch = img_tensor.unsqueeze(dim= 0)
        
        self.model.eval()
        with torch.inference_mode():
            
            predict_logit = self.model(img_tensor_in_batch)
            soft_max_persent = torch.softmax(predict_logit, dim= 1)
            predicted_class = self.class_names[torch.argmax(predict_logit, dim= 1)]

            score = (soft_max_persent[0, torch.argmax(predict_logit, dim= 1)].item())

        return predict_logit, predicted_class, score

    async def predict(self, img: Image):
        """
        Args:
            img: PIL.Image()
     
        Returns:
            dict: {
                "predicted_class" : int,
                "score" : float,
            }
        """
        
        _, predicted_class, score = self._predict(img=img)

        return_dict = {
            "predicted_class" : predicted_class,
            "score" : score,
        }
        return return_dict
