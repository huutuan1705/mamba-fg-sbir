import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
        self.norm = nn.LayerNorm(2048)
        self.mha = nn.MultiheadAttention(2048, num_heads=args.num_heads, batch_first=True)
    
    def fix_weights(self):
        for x in self.parameters():
            x.requires_grad = False
            
    def forward(self, x):
        bs, c, h, w = x.shape
        x_att = x.reshape(bs, c, h*w).transpose(1, 2)
        x_att = self.norm(x_att)
        att_out, _  = self.mha(x_att, x_att, x_att)
        att_out = att_out.transpose(1, 2).reshape(bs, c, h, w)
        att_out = self.pool_method(att_out).view(-1, 2048)
        return F.normalize(att_out)
        
class Linear_global(nn.Module):
    def __init__(self, feature_num):
        super(Linear_global, self).__init__()
        self.head_layer = nn.Linear(2048, feature_num)
    
    def fix_weights(self):
        for x in self.parameters():
            x.requires_grad = False
            
    def forward(self, x):
        return F.normalize(self.head_layer(x))