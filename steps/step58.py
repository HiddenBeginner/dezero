'''
# Step58 대표적인 CNN(VGG16)
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import dezero
import dezero.functions as F
from dezero import Variable
from dezero.models import VGG16

if __name__ == '__main__':
    # 58.2 학습된 가중치 데이터
    model = VGG16(pretrained=True)
    x = np.random.randn(1, 3, 244, 244).astype(np.float32)

    # 58.3 학습된 VGG16 사용하기
    url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
    img_path = dezero.utils.get_file(url)
    img = Image.open(img_path)
    # img.show()

    x = VGG16.preprocess(img)
    x = x[np.newaxis]
    with dezero.test_mode():
        y = model(x)
    predict_id = np.argmax(y.data)
    labels = dezero.datasets.ImageNet.labels()
    print(labels[predict_id])
