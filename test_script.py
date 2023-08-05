import os

if __name__ == '__main__':
    for var in range(1, 5):
        os.system(f'python test_nwks.py --net resnet8 --nway 7 --kshot {var}')
        os.system(f'python test_nwks.py --net resnet12 --nway 7 --kshot {var}')
    for var in range(1, 5):
        os.system(f'python test_nwks.py --net resnet8 --nway 40 --kshot {var}')
        os.system(f'python test_nwks.py --net resnet12 --nway 40 --kshot {var}')
