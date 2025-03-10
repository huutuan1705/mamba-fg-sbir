import torch
import argparse
import torch.utils.data as data 

from tqdm import tqdm
from dataset.dataset import FGSBIR_Dataset
from model import Mamba_FGSBIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    dataset_train = FGSBIR_Dataset(args, mode='train', on_fly=True)
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))
    
    dataset_test = FGSBIR_Dataset(args, mode='test', on_fly=True)
    dataloader_test = data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))
    
    return dataloader_train, dataloader_test


def load_pretrained(model, args):
    if args.load_pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    
    return model

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Mamba Fine-Grained SBIR model')
    parsers = argparse.ArgumentParser(description='Base Fine-Grained SBIR model')
    parsers.add_argument('--dataset_name', type=str, default='ShoeV2')
    parsers.add_argument('--backbone_name', type=str, default='InceptionV3', help='VGG16/InceptionV3/ResNet50')
    parsers.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parsers.add_argument('--output_size', type=int, default=64)
    parsers.add_argument('--d_model', type=int, default=2048, help='Hidden of dimension size')
    parsers.add_argument('--num_heads', type=int, default=8, help='number of head self attention')
    parsers.add_argument('--n_layers', type=int, default=2, help='Number of mamba layer')
    parsers.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for head')
    parsers.add_argument('--root_dir', type=str, default='./../')
    
    parsers.add_argument('--pretrained_dir', type=str, default='./../')
    parsers.add_argument('--load_pretrained_best', type=bool, default=False)
    parsers.add_argument('--load_pretrained_last', type=bool, default=False)
    parsers.add_argument('--use_kaiming_init', type=bool, default=True)
    
    parsers.add_argument('--batch_size', type=int, default=16)
    parsers.add_argument('--test_batch_size', type=int, default=1)
    parsers.add_argument('--step_size', type=int, default=100)
    parsers.add_argument('--num_anchors', type=int, default=25)
    parsers.add_argument('--gamma', type=float, default=0.5)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--learning_rate', type=float, default=0.001)
    parsers.add_argument('--epochs', type=int, default=200)
    parsers.add_argument('--eval_freq_iter', type=int, default=100)
    parsers.add_argument('--print_freq_iter', type=int, default=1)
    
    args = parsers.parse_args()
    dataloader_train, dataloader_test = get_dataloader(args=args)
    
    model = Mamba_FGSBIR(args=args)
    model.to(device)
    
    if args.load_pretrained_best:
        model.load_state_dict(torch.load(args.pretrained_dir + "/" + args.dataset_name + '_best.pth'))
    
    if args.load_pretrained_best==False:
        backbones_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_backbone.pth")
        attention_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_attention.pth")
        linear_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_linear.pth")
    
        model.sample_embedding_network.load_state_dict(backbones_state['sample_embedding_network'])
        model.attention.load_state_dict(attention_state['attention'])
        model.linear.load_state_dict(linear_state['linear'])
        model.mamba_linear.load_state_dict(linear_state['linear'])
        
        # model.sketch_embedding_network.load_state_dict(backbones_state['sketch_embedding_network'])
        # model.sketch_attention.load_state_dict(attention_state['sketch_attention'])
        # model.sketch_linear.load_state_dict(linear_state['sketch_linear'])
        
        step_count, top1, top5, top10, meanA, meanB = -1, 0, 0, 0, 0, 0
        for i_epoch in range(args.epochs):
            print(f"Epoch: {i_epoch+1} / {args.epochs}")
            loss = 0
            for _, batch_data in enumerate(tqdm(dataloader_train)):
                step_count = step_count + 1
                model.train()
                loss = model.train_model(batch=batch_data)
            
            with torch.no_grad():
                model.eval()
                top1_eval, top5_eval, top10_eval, meanA_eval, meanB_eval = model.evaluate(dataloader_test)
                
                if top10_eval > top10:
                    top1, top5, top10 = top1_eval, top5_eval, top10_eval
                    meanA, meanB = meanA_eval, meanB_eval
                    torch.save(model.state_dict(), args.dataset_name + '_best.pth')
                
                torch.save(model.state_dict(), args.dataset_name + '_last.pth')
                    
            print('Top 1 accuracy:  {:.4f}'.format(top1_eval))
            print('Top 5 accuracy:  {:.4f}'.format(top5_eval))
            print('Top 10 accuracy: {:.4f}'.format(top10_eval))
            print('Mean A:          {:.4f}'.format(meanA_eval))
            print('Mean B:          {:.4f}'.format(meanB_eval))
            print('Loss:            {:.4f}'.format(loss))
            print("========================================")
