import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from tqdm import tqdm

from backbone import InceptionV3
from attention import Attention_global, Linear_global

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Model(nn.Module):
    def __init__(self, args):
        super(FGSBIR_Model, self).__init__()
        self.sample_embedding_network = eval(args.backbone_name + "(args)")
        self.sketch_embedding_network = eval(args.backbone_name + "(args)")
        self.loss = nn.TripletMarginLoss(margin=args.margin)        
        self.sample_train_params = self.sample_embedding_network.parameters()
        self.sketch_train_params = self.sketch_embedding_network.parameters()
        self.args = args
        
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
        
        self.attention = Attention_global()
        self.attn_params = self.attention.parameters()
        
        self.sketch_attention = Attention_global()
        self.sketch_attn_params = self.sketch_attention.parameters()
        
        self.linear = Linear_global(feature_num=self.args.output_size)
        self.linear_params = self.linear.parameters()
        
        self.sketch_linear = Linear_global(feature_num=self.args.output_size)
        self.sketch_linear_params = self.sketch_linear.parameters()

        if self.args.use_kaiming_init:
            self.attention.apply(init_weights)
            self.sketch_attention.apply(init_weights)
            self.linear.apply(init_weights)
            self.sketch_linear.apply(init_weights)
            
        self.optimizer = optim.Adam([
            {'params': self.sketch_embedding_network.parameters(), 'lr': args.learning_rate},
            {'params': self.sample_embedding_network.parameters(), 'lr': args.learning_rate},
            {'params': self.attention.parameters(), 'lr': args.lr_att_linear},
            {'params': self.sketch_attention.parameters(), 'lr': args.lr_att_linear},
            {'params': self.linear.parameters(), 'lr': args.lr_att_linear},
            {'params': self.sketch_linear.parameters(), 'lr': args.lr_att_linear},
        ])
        
        # self.optimizer = optim.Adam(self.sample_train_params, self.args.learning_rate)
        # self.optimizer = optim.Adam(self.sketch_train_params, self.args.learning_rate)
        # self.optimizer = optim.Adam([
        #     {'params': filter(lambda param: param.requires_grad, self.sample_train_params), 'lr': self.args.learning_rate},
        #     {'params': filter(lambda param: param.requires_grad, self.sketch_train_params), 'lr': self.args.learning_rate},
        #     {'params': self.attn_params, 'lr': self.args.learning_rate},
        #     {'params': self.linear_params, 'lr': self.args.learning_rate},
        #     {'params': self.sketch_attn_params, 'lr': self.args.learning_rate},
        #     {'params': self.sketch_linear_params, 'lr': self.args.learning_rate},])
        
    def test_forward(self, batch):
        sketch_feature = self.sketch_embedding_network(batch['sketch_img'].to(device))
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        
        if self.args.use_attention:
            positive_feature = self.attention(positive_feature)
            sketch_feature = self.sketch_attention(sketch_feature)
        
        if self.args.use_linear:
            positive_feature = self.linear(positive_feature)
            sketch_feature = self.sketch_linear(sketch_feature)
            
        return sketch_feature, positive_feature
        
    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
            
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        negative_feature = self.sample_embedding_network(batch['negative_img'].to(device))
        sketch_feature = self.sketch_embedding_network(batch['sketch_img'].to(device))
        
        if self.args.use_attention:
            positive_feature = self.attention(positive_feature)
            negative_feature = self.attention(negative_feature)
            sketch_feature = self.sketch_attention(sketch_feature)
            
        if self.args.use_linear:
            positive_feature = self.linear(positive_feature)
            negative_feature = self.linear(negative_feature)
            sketch_feature = self.sketch_linear(sketch_feature)

        loss = self.loss(sketch_feature, positive_feature, negative_feature)
        loss.backward()
        self.optimizer.step()

        return loss.item() 

    def evaluate(self, datloader_test):
        Image_Feature_ALL = []
        Image_Name = []
        Sketch_Feature_ALL = []
        Sketch_Name = []
        start_time = time.time()
        self.eval()
        for i_batch, sanpled_batch in enumerate(tqdm(datloader_test)):
            sketch_feature, positive_feature= self.test_forward(sanpled_batch)
            Sketch_Feature_ALL.extend(sketch_feature)
            Sketch_Name.extend(sanpled_batch['sketch_path'])

            for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
                if positive_name not in Image_Name:
                    Image_Name.append(sanpled_batch['positive_path'][i_num])
                    Image_Feature_ALL.append(positive_feature[i_num])

        rank = torch.zeros(len(Sketch_Name))
        Image_Feature_ALL = torch.stack(Image_Feature_ALL)

        for num, sketch_feature in enumerate(Sketch_Feature_ALL):
            s_name = Sketch_Name[num]
            sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
            position_query = Image_Name.index(sketch_query_name)

            # print("sketch_feature.unsqueeze(0) shape: ", sketch_feature.unsqueeze(0).shape) #[1, 64]
            # print("Image_Feature_ALL shape: ", Image_Feature_ALL.shape) #[200, 64]
            # print("Image_Feature_ALL[position_query].unsqueeze(0) shape: ", Image_Feature_ALL[position_query].unsqueeze(0).shape) # [1, 64]
            
            distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
            target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                  Image_Feature_ALL[position_query].unsqueeze(0))

            rank[num] = distance.le(target_distance).sum()

        top1 = rank.le(1).sum().numpy() / rank.shape[0]
        top5 = rank.le(5).sum().numpy() / rank.shape[0]
        top10 = rank.le(10).sum().numpy() / rank.shape[0]

        # print('Time to EValuate:{}'.format(time.time() - start_time))
        return top1, top5, top10