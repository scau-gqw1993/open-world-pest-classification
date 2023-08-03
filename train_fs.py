import torch
import torch.optim as optim
import torch.nn as nn
import os

import sample
import util
import fewcls


def train(net, net_name, optimizer, criterion, data_dir, itera_size, n_way, k_shot, q_query):
    net.train()
    criterion = criterion.cuda()
    net = net.cuda()

    dataset = util.getfilepath(data_dir)

    for it in range(itera_size):
        tmp = sample.sample(dataset, n_way, k_shot, q_query)
        x_centers = []
        x_queries = []
        for v1 in tmp:
            x_shots = []
            for v2 in v1[:k_shot]:
                x_shots.append(util.process_image(v2, 'train'))
            x_shots = torch.stack(x_shots).cuda()
            x_shots = net(x_shots)
            x_centers.append(torch.mean(x_shots, dim=0, keepdim=True))

            for v2 in v1[k_shot:]:
                x_queries.append(util.process_image(v2, 'train'))

        x_centers = torch.cat(x_centers)

        x_queries = torch.stack(x_queries).cuda()
        x_queries = net(x_queries)

        x_centers = x_centers.cuda()
        x_queries = x_queries.cuda()

        outputs = fewcls.compute_similar(x_centers, x_queries)

        y_labels = []
        for i in range(n_way):
            for _ in range(q_query):
                y_labels.append(i)
        y_labels = torch.Tensor(y_labels).long().cuda()

        loss = criterion(outputs, y_labels)
        if it % 50 == 0:
            print(loss.item())
            os.makedirs('./save', exist_ok=True)
            model_out_path = os.path.join('./save/{}.pth'.format(net_name))
            torch.save(net.state_dict(), model_out_path)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        os.makedirs('./save', exist_ok=True)
        model_out_path = os.path.join('./save/{}.pth'.format(net_name))
        torch.save(net.state_dict(), model_out_path)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='resnet12', help='network name')
    arg = parser.parse_args()

    net_name = arg.net
    net = None

    if net_name == 'resnet12':
        from models import ResNet12
        net = ResNet12.resnet12()
    elif net_name == 'resnet18':
        from models import ResNet18
        net = ResNet18.resnet18()
    elif net_name == 'resnet34':
        from models import ResNet34
        net = ResNet34.resnet34()
    elif net_name == 'resnet50':
        from models import ResNet50
        net = ResNet50.resnet50()
    elif net_name == 'resnest':
        from models import ResNeSt
        net = ResNeSt.resnest()
    elif net_name == 'sanet':
        from models import SANet
        net = SANet.mbv2_sa()
    elif net_name == 'mobilenetxt':
        from models import MobileNetXt
        net = MobileNetXt.mobilenext()
    elif net_name == 'tripletnet':
        from models import TripletNet
        net = TripletNet.mbv2_triplet()
    elif net_name == 'lcnet':
        from models import LCNet
        net = LCNet.lcnet_baseline()
    elif net_name == 'ghostnet':
        from models import GhostNet
        net = GhostNet.ghostnet()
    elif net_name == 'canet':
        from models import CANet
        net = CANet.mbv2_ca()
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

    optimizer = optim.Adam((net.parameters()), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    train(net, net_name, optimizer, criterion, './dataset/train/', 8001, 7, 5, 5)
