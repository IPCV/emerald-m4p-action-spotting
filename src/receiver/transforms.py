import io

import torchvision.transforms as transforms
from PIL import Image


def transform_image(image):
    my_transforms = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])

    return my_transforms(image).unsqueeze(0)
