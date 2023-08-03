import os

if __name__ == "__main__":
    '''
    name_list = ['resnet12', 'mobilenetv1', 'mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_large', 'squeezenet',
                 'xception', 'inceptionv2', 'inceptionv3', 'inceptionv4', 'efficientnet_b0', 'efficientnet_b1',
                 'shufflenet', 'shufflenetv2', 'mnasnet']
    '''
    name_list = ['resnet8', 'inceptionv1']
    for var in name_list:
        os.system(f'python test_all.py --net {var}')
