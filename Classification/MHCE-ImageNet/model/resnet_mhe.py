import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np

def get_groups(n_labels, num_group, num_ele_per_group):
    group_y , idx = [] , 0
    for i in range(num_group-1):
        group_y.append(np.array([idx + _ for _ in range(num_ele_per_group)]))
        idx += num_ele_per_group
    has_padding = False
    if n_labels - idx>0:
        last_y = np.array([idx + _ for _ in range(n_labels -idx)])
        if len(last_y)< num_ele_per_group:
            num_pad = num_ele_per_group - len(last_y)
            last_y = np.pad(last_y,(0, num_pad), 'constant', constant_values=n_labels)
            has_padding = True
            print('*** pad last group with constant mode, length = ',num_pad)
        group_y.append(last_y)
    assert len(group_y) == num_group and len(group_y[-1])==num_ele_per_group
    return torch.tensor(group_y)#, has_padding

def get_one_hot(lables, num_cls):
    one_hot = torch.zeros((len(lables),num_cls), device='cuda')
    one_hot.scatter_(1, lables.view(-1, 1).long(), 1)
    return one_hot

class ResNet(torch.nn.Module):
    def __init__(self,num_classes, name, hidden_dim=256,*args, **kwargs):
        super(ResNet, self).__init__()
        if name == 'resnet18':
            resnet = models.resnet18(pretrained=False)
            feature_dim = 512
        elif name == 'resnet50':
            resnet = models.resnet50(pretrained=False)
            feature_dim = 2048

        self.net = torch.nn.Sequential(*list(resnet.children())[:-1])
        
        self.num_classes = num_classes
        num_labels = num_classes[0] * num_classes[1]
        self.group = nn.Linear(feature_dim, num_classes[0])
        self.classifier = nn.Linear(feature_dim, hidden_dim)
        self.embed = nn.Embedding(num_labels, hidden_dim)
        nn.init.xavier_uniform_(self.embed.weight)
        print('resnet mhe ...',name)
        self.group_y = get_groups(num_labels, num_classes[0], num_classes[1])
        print(self.group_y.shape,'gorup y')
        self.loss_fn = nn.CrossEntropyLoss()
        

    def get_candidates(self, g_out, g_label=None):
        outputs = torch.sigmoid(g_out.detach())
        if g_label is not None:
            outputs += get_one_hot(g_label,self.num_classes[0])
        scores, indices = torch.topk(outputs, k=1)
        topk = self.group_y[indices].squeeze(1)
        scores = scores.repeat(1,self.num_classes[1])
        return topk, scores

    def forward(self, x, label=None):
        h = self.net(x)
        h = h.view(-1, h.shape[1])
        
        g_out=self.group(h)
        if label is not None:
            g_label = label//self.num_classes[1]
            local_label = label % self.num_classes[1]
            candidates, _ = self.get_candidates(g_out, g_label)
            local_logit = self._get_embed_outputs(candidates,h)
            loss_l = self.loss_fn(local_logit, local_label)
            loss_g = self.loss_fn(g_out, g_label)
            pred = torch.argmax(local_logit.data,-1).eq(local_label.data)
            g_pred = torch.argmax(g_out.data,-1).eq(g_label.data)
            return loss_l+loss_g, pred, g_pred
        else:
            candidates, scores = self.get_candidates(g_out)
            outputs = self._get_embed_outputs(candidates, h)
            outputs = torch.sigmoid(outputs) * scores.to(outputs.device)
            return outputs, g_out
    
    def _get_embed_outputs(self, candidates, features):
        emb = self.classifier(features)
        embed_weights = self.embed(candidates.to(features.device))
        outputs = torch.bmm(embed_weights, emb.unsqueeze(-1)).squeeze(-1)
        return outputs

if __name__=='__main__':
    x = torch.rand((2,3,224,224)).cuda()
    t = torch.randint(0,12,(2,)).cuda()
    net = ResNet([300,350],name='resnet50').cuda()
    outputs = net(x,t)
    
    print(outputs.shape)