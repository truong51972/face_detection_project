import torch
from torch import nn
from torchvision import transforms

from pathlib import Path
from torchmetrics import Accuracy, ConfusionMatrix
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecall

from pytorch_packages.data_setup import create_dataloader
from pytorch_packages.engine import train, test
from pytorch_packages.utils import plot_save_model
from pytorch_packages.model_builder import load_model

def run(**kwargs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: '{device}'")
    
    torch.cuda.empty_cache()
    img_size = 96
    train_transforms_data = transforms.Compose([
        transforms.Resize(size= (img_size, img_size)),
        # transforms.RandomResizedCrop(size=(img_size, img_size), antialias=True),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
        transforms.transforms.ColorJitter(brightness= [0.9, 1.1], contrast= [0.9, 1.1]),
        transforms.ToTensor(),
    ])
    
    val_transforms_data = transforms.Compose([
        transforms.Resize(size= (img_size, img_size)),
        transforms.ToTensor(),
    ])

    test_transforms_data = transforms.Compose([
        transforms.Resize(size= (img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    train_dataloader, val_dataloader, test_dataloader, class_names = create_dataloader(train_transform=train_transforms_data,
                                                                                      val_transform=val_transforms_data,
                                                                                      test_transform=test_transforms_data,
                                                                                      **kwargs)

    model_name = kwargs['model']
    pretrain_model_path= kwargs['train_para']['pretrain_model_path']
    model, info_data = load_model(model_name= model_name, class_names= class_names, pretrain_model_path= pretrain_model_path, device= device)
    
    loss_func = nn.CrossEntropyLoss()

    lr = kwargs['train_para']['optimize']['learning_rate']
    momentum = kwargs['train_para']['optimize']['momentum']
    weight_decay = kwargs['train_para']['optimize']['weight_decay']
    optimizer = torch.optim.SGD(params= model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    mectric_funcs = []
    mectric_funcs.append(Accuracy(task='multiclass', num_classes= len(class_names)).to(device))
    mectric_funcs.append(MulticlassPrecision(num_classes= len(class_names)).to(device))
    mectric_funcs.append(MulticlassRecall(num_classes= len(class_names)).to(device))

    step_size = kwargs['train_para']['lr_scheduler']['step_size']
    gamma = kwargs['train_para']['lr_scheduler']['gamma']
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size= step_size,
        gamma= gamma
    )

    epochs= kwargs['train_para']['epochs']
    save_checkpoint_freq= kwargs['train_para']['save_checkpoint_freq']
    verbose= kwargs['train_para']['verbose']
    
    results = train(
            model= model,
            train_dataloader= train_dataloader,
            val_dataloader= val_dataloader,
            loss_func= loss_func,
            optimizer= optimizer,
            lr_scheduler= lr_scheduler,
            mectric_funcs= mectric_funcs,
            epochs= epochs,
            info_data = info_data,
            save_checkpoint_freq= save_checkpoint_freq,
            verbose= verbose,
            device= device,
    )

    results = test(model=model,
                  test_dataloader=test_dataloader,
                  results=results,
                  verbose= verbose,
                  device= device)

    is_save = True if epochs > 0 else False
    plot_save_model(
        model= model,
        model_name= model_name,
        results= results,
        class_names= class_names,
        is_save= is_save,
        device= device
    )
    
    torch.cuda.empty_cache()

if __name__ == '__main__':
    config = {
        'dataset' : {
            'train': './.datasets/splitted_facial_emotion/train/',
            'val':   './.datasets/splitted_facial_emotion/val/',    
            'test':  './.datasets/splitted_facial_emotion/test/',
            'batch_size': 32,
        },
        'model': 'vgg11_bn',
        'train_para': {
            'pretrain_model_path': None,
            'epochs': 40,
            'optimize' : {
                'learning_rate': 0.001,
                'momentum': 0.9,
                'weight_decay': 0.00001,
            },
            'lr_scheduler' : {
                'step_size' : 5,
                'gamma' : 0.1,
            },
            'save_checkpoint_freq' : 0,
            'verbose': True
        },
    }

    # train.run(**config)
    run(**config)