import torchvision.models as models
import torch
import torch.nn as nn


class ResNet(torch.nn.Module):
    def __init__(self,num_classes, name,*args, **kwargs):
        super(ResNet, self).__init__()
        if name == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif name == 'resnet50':
            resnet = models.resnet50(pretrained=False)

        self.net = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(2048 , sum(num_classes))

    def forward(self, x):
        h = self.net(x)
        h = h.view(-1, h.shape[1])
        return self.classifier(h)

if __name__=='__main__':
    x = torch.rand((2,3,224,224))
    net = ResNet([300,350],name='resnet50')
    outputs = net(x)
    
    print(outputs.shape)