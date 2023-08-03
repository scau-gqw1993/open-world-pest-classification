import torch
from tqdm import tqdm

from models import ResNet8
import fewcls

if __name__ == '__main__':
    model = ResNet8.resnet8().eval()

    centers = torch.ones([40, 512])

    for _ in tqdm(range(100)):
        x = torch.ones([40, 3, 112, 112])
        y = model(x)

        outputs = fewcls.compute_similar(centers, y)
        preds = torch.argmax(outputs, dim=-1)
        preds = preds.numpy()
