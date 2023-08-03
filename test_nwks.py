import torch
from tqdm import tqdm
import os
import numpy as np
from scipy import stats

import sample
import util
import fewcls


def examine(net, data_dir, num_sample, n_way, k_shot, q_query):
    net.eval()
    correct = 0

    dataset = util.getfilepath(data_dir)
    with torch.no_grad():
        for _ in tqdm(range(num_sample), ncols=50):
            tmp = sample.sample(dataset, n_way, k_shot, q_query)
            x_centers = []
            x_queries = []
            for v1 in tmp:
                x_shots = []
                for v2 in v1[:k_shot]:
                    x_shots.append(util.process_image(v2, 'test'))
                x_shots = torch.stack(x_shots)
                x_shots = net(x_shots)
                x_centers.append(torch.mean(x_shots, dim=0, keepdim=True))

                for v2 in v1[k_shot:]:
                    x_queries.append(util.process_image(v2, 'test'))

            x_centers = torch.cat(x_centers)

            x_queries = torch.stack(x_queries)
            x_queries = net(x_queries)

            outputs = fewcls.compute_similar(x_centers, x_queries)

            y_labels = []
            for i in range(n_way):
                for _ in range(q_query):
                    y_labels.append(i)

            preds = torch.argmax(outputs, dim=-1)
            preds = preds.numpy()
            correct += np.sum(np.array(y_labels) == preds)

    res = correct / (num_sample * n_way * q_query)

    return res


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='resnet12', help='network name')
    parser.add_argument('--nway', type=int, default=7, help='support set - n ways')
    parser.add_argument('--kshot', type=int, default=5, help='support set - k shots')
    arg = parser.parse_args()

    net_name = arg.net
    net = None

    if net_name == 'resnet12':
        from models import ResNet12
        net = ResNet12.resnet12()
    elif net_name == 'resnet8':
        from models import ResNet8
        net = ResNet8.resnet8()
    elif net_name == 'mobilenetv1':
        from models import MobileNetV1
        net = MobileNetV1.MobileNetV1()
    elif net_name == 'mobilenetv2':
        from models import MobileNetV2
        net = MobileNetV2.MobileNetV2()
    elif net_name == 'mobilenetv3_small':
        from models import MobileNetV3
        net = MobileNetV3.MobileNetV3_Small()
    elif net_name == 'mobilenetv3_large':
        from models import MobileNetV3
        net = MobileNetV3.MobileNetV3_Large()
    elif net_name == 'squeezenet':
        from models import SqueezeNet
        net = SqueezeNet.SqueezeNet()
    elif net_name == 'xception':
        from models import Xception
        net = Xception.Xception()
    elif net_name == 'inceptionv1':
        from models import InceptionV1
        net = InceptionV1.InceptionV1()
    elif net_name == 'inceptionv2':
        from models import InceptionV2
        net = InceptionV2.InceptionV2()
    elif net_name == 'inceptionv3':
        from models import InceptionV3
        net = InceptionV3.InceptionV3()
    elif net_name == 'inceptionv4':
        from models import InceptionV4
        net = InceptionV4.InceptionV4()
    elif net_name == 'efficientnet_b0':
        from models import EfficientNet
        net = EfficientNet.EfficientNet('efficientnet_b0')
    elif net_name == 'efficientnet_b1':
        from models import EfficientNet
        net = EfficientNet.EfficientNet('efficientnet_b1')
    elif net_name == 'shufflenet':
        from models import ShuffleNet
        net = ShuffleNet.ShuffleNet()
    elif net_name == 'shufflenetv2':
        from models import ShuffleNetV2
        net = ShuffleNetV2.ShuffleNetV2()
    elif net_name == 'mnasnet':
        from models import MnasNet
        net = MnasNet.MnasNet()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(os.path.join('./save/{}.pth'.format(net_name)), map_location=device)
    net.load_state_dict(state_dict)

    accs = []

    for i in range(10):
        acc = examine(net, './dataset/test', 100, arg.nway, arg.kshot, 1)
        print(acc)
        accs.append(acc)

    avg = np.mean(accs)
    std = stats.sem(accs)

    print(net_name)
    print(avg, std, 1.96 * std)

    with open(f'{arg.nway}way{arg.kshot}shot_results.txt', 'a') as tf:
        tf.write(f'{net_name}\t{avg}\t{1.96 * std}\n')
