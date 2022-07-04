import torch
from torch import nn
import utils

class XMLModel(nn.Module):
    def __init__(self, n_labels, num_classes=[86,46], model_name='bert-base',
                 model_path = '', candidates_topk=10, hidden_dim=300, 
                 feature_layers=5, dropout=0.5, device='cuda'):
        super(XMLModel, self).__init__()

        self.swa_state = {}
        self.candidates_topk = candidates_topk

        self.model = utils.get_model(model_name, path = model_path)
        self.feature_layers= feature_layers
        len_feature = self.feature_layers*self.model.config.hidden_size
        self.drop_out = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(len_feature)
        self.group_classifer = nn.Linear(len_feature, num_classes[0])
        self.sub_classifer = nn.Linear(len_feature, hidden_dim)
        
        
        self.group_y, has_padding = utils.get_groups(n_labels, num_classes[0], num_classes[1])
        self.embedding = nn.Embedding(n_labels+1 if has_padding else n_labels, hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.device=device
        
        self.group_y = self.group_y.to(device)
        print(self.group_y[-2],len(self.group_y[-2]))
        print(self.group_y[-1],len(self.group_y[-1]))
        print('has_padding===: ', has_padding)
    

    def forward(self, input_ids, attention_mask, token_type_ids, targets=None, group_labels=None):

        outs = self.model(input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)[-1]
        features = torch.cat([outs[-i][:, 0] for i in range(1, self.feature_layers+1)], dim=-1)
        #features = self.drop_out(features)
        group_outputs = self.group_classifer(features)
        
        if targets is not None:
            candidates, _ = utils.get_topk_candidates(self.group_y, group_outputs,
                                                        group_labels=group_labels,
                                                        topk = self.candidates_topk)
            new_labels = utils.get_candidate_labels(targets, candidates)
            #new_labels, candidates = utils.get_element_labels_v2(targets, candidates)
            outputs = self._get_embed_outputs(candidates,features)
            loss = self.loss_fn(outputs, new_labels.to(self.device)) \
                    + self.loss_fn(group_outputs, group_labels)
            return outputs, loss
        else:
            candidates, scores = utils.get_topk_candidates(self.group_y, group_outputs,
                                                            topk = self.candidates_topk)
            outputs = self._get_embed_outputs(candidates,features)
            outputs = torch.sigmoid(outputs) * scores.to(self.device)
            return outputs, candidates, group_outputs

    def _get_embed_outputs(self,candidates,features):
        emb = self.sub_classifer(features)
        embed_weights = self.embedding(candidates)
        outputs = torch.bmm(embed_weights, emb.unsqueeze(-1)).squeeze(-1)
        return outputs

    def swa_init(self):
        self.swa_state = {'models_num': 1}
        for n, p in self.named_parameters():
            self.swa_state[n] = p.data.cpu().clone().detach()

    def swa_step(self):
        if 'models_num' not in self.swa_state: return
        self.swa_state['models_num'] += 1
        beta = 1.0 / self.swa_state['models_num']
        with torch.no_grad():
            for n, p in self.named_parameters():
                self.swa_state[n].mul_(1.0 - beta).add_(p.data.cpu(), alpha=beta)

    def swa_swap_params(self):
        if 'models_num' not in self.swa_state:
            return
        for n, p in self.named_parameters():
            self.swa_state[n], p.data =  self.swa_state[n].cpu(), p.data.cpu()
            self.swa_state[n], p.data =  p.data.cpu(), self.swa_state[n].cuda()