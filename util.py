import os
from torchvision import transforms
from PIL import Image


input_size = 112


def getfilepath(dirpath):
    res = []

    species = os.listdir(dirpath)
    for var in species:
        tmp1 = os.listdir(os.path.join(dirpath, var))
        tmp2 = []
        for t1 in tmp1:
            tmp2.append(os.path.join(dirpath, var, t1))
        res.append(tmp2)

    return res


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomResizedCrop(size=input_size, scale=(0.7, 1)),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}

def process_image(image_path, setname):
    image = Image.open(image_path)
    image = data_transforms[setname](image)
    return image

if __name__ == '__main__':
    res = getfilepath('')
    image_path = res[0][0]
    v1 = process_image(image_path, 'train')
    image_path = res[3][1]
    v2 = process_image(image_path, 'train')
    print(type(v1), type(v2))

    import torch

    v3 = torch.stack((v1, v2, v1, v2))

    from models import resnet12

    net = resnet12.resnet12()
    y = net(v3)

    print(y.shape)
