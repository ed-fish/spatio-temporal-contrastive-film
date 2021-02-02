from torchvision import models

def return_resnet():
    resnet = models.video.r3d_18(pretrained=True)