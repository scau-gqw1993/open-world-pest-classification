import os

if __name__ == "__main__":
    '''
    name_list = ['resnet12', 'mobilenetv1', 'mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_large', 'squeezenet',
                 'xception', 'inceptionv2', 'inceptionv3', 'inceptionv4', 'efficientnet_b0', 'efficientnet_b1',
                 'shufflenet', 'shufflenetv2', 'mnasnet']
    '''
    name_list = ['resnet50', 'resnet34', 'resnet18', 'mobilenetxt', 'addernet', 'ghostnet',
                 'canet', 'resnest', 'sanet', 'tripletnet', 'lcnet']
    for var in name_list:
        os.system(f'python train_fs.py --net {var}')

