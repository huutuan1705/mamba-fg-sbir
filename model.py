import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from tqdm import tqdm

from base_fg_sbir.backbone import InceptionV3
from base_fg_sbir.attention import Linear_global, SelfAttention
from mamba_block.model import MambaModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mamba_FGSBIR(nn.Module):
    def __init__(self, args):
        super(Mamba_FGSBIR, self).__init__()
        self.sample_embedding_network = eval(args.backbone_name + "(args)")
        self.loss = nn.TripletMarginLoss(margin=args.margin)        
        self.sample_train_params = self.sample_embedding_network.parameters()
        self.args = args
        
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
                
        self.attention = SelfAttention()
        self.linear = Linear_global(feature_num=self.args.output_size)
        self.mamba = MambaModule()
        
        self.attention.fix_weights()
        self.linear.fix_weights()
        
        if self.args.use_kaiming_init:
            self.mamba.apply(init_weights)
            
        self.optimizer = optim.Adam([
            {'params': self.mamba.parameters(), 'lr': args.lr_att_linear},
        ])
        
    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        negative_feature = self.sample_embedding_network(batch['negative_img'].to(device))
        
        positive_feature = self.linear(self.attention(positive_feature)) # (N, 64)
        negative_feature = self.linear(self.attention(negative_feature)) # (N, 64)
        
        sketch_tensors = batch['sketch_imgs'] # (N, 25, 3, 299, 299)
        sketch_features = []
        for i in range(sketch_tensors.shape[0]):
            sketch_feature = sketch_tensors[i]
            sketch_feature = self.sample_embedding_network(self.attention(sketch_feature))
            sketch_features.append(sketch_feature)
        
        sketch_features = torch.stack(sketch_features) # (N, 25, 2048)
        
        sketch_feature = self.mamba(sketch_features) # (N, 64)
        loss = self.loss(sketch_feature, positive_feature, negative_feature)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def test_forward(self, batch):
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        positive_feature = self.linear(self.attention(positive_feature))
        
        sketch_tensors = batch['sketch_imgs'] # (N, 25, 3, 299, 299)
        sketch_features = []
        for i in range(sketch_tensors.shape[0]):
            sketch_feature = sketch_tensors[i]
            sketch_feature = self.sample_embedding_network(self.attention(sketch_feature))
            sketch_features.append(sketch_feature)
        
        sketch_features = torch.stack(sketch_features) # (N, 25, 2048)
        
        return positive_feature, sketch_features
    
    def evaluate(self, datloader_test):
        self.eval()
        
        image_features_all = []
        image_name = []
        sketch_features_all = []
        sketch_name = []
        
        for _, sample_batch in enumerate(tqdm(datloader_test)):
            positive_feature, sketch_features = self.test_forward(sample_batch)
            sketch_features_all.append()
        