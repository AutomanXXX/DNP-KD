import torch
from dataset import get_data_transforms
import numpy as np
import random
import os
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from dataset import MVTecDataset_train,VisaDataset_train
import argparse
from tqdm import tqdm
from Subnet_w_fftmask import SubNet_w_FFTMask
import sys


def write_loss_to_file(epoch, epochs, loss_re,_class_,model_name):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')
    fin_str = _class_+ " | " + "epoch "+ "["+ str(epoch) + "/" + str(epochs) +"], " + "loss_re:"+ str(loss_re)
    fin_str += "\n"
    path = "./outputs/"+model_name+"_results.txt"
    with open(path,'a+') as file:
        file.write(fin_str)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        predictions_tensor = predictions 
        if isinstance(targets, list):
            targets = torch.tensor(targets, dtype=torch.long)
        loss = self.criterion(predictions_tensor, targets)
        return loss

def train(model_name):
    epochs = 50
    learning_rate = 0.005
    batch_size = 8
    image_size = 256

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs("checkpoints",exist_ok=True)
  
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    
    ckp_path = './checkpoints/' + model_name+ '_Unified_'

    if args.dataset=='mvtec':
        train_path ='./mvtecdataset/*/train'
        train_data = MVTecDataset_train(root=train_path,transform=data_transform)

    else:
        train_path ='./visadataset/*/train'
        train_data = VisaDataset_train(root=train_path,transform=data_transform)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    if args.dataset=='mvtec':
        subnet_w_fftmask = SubNet_w_FFTMask(num_classes=15)
    else:
        subnet_w_fftmask = SubNet_w_FFTMask(num_classes=12)
    subnet_w_fftmask = subnet_w_fftmask.to(device)


    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_fn = CustomLoss()

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters())+list(subnet_w_fftmask.parameters()), lr=learning_rate, betas=(0.5,0.999))

    for epoch in range(epochs):

        bn.train()
        decoder.train()

        loss_re_list = []
        total_batches = len(train_dataloader)
        loop = tqdm((train_dataloader), total = total_batches)

        for img, label, class_labels in loop:

            class_labels = class_labels.to(device)
            img = img.to(device)
            inputs = encoder(img)
            predicts, masks, out_subnet, out_subnet_con = subnet_w_fftmask(inputs, train = True)
            mid = bn(out_subnet)
            outputs = decoder(mid)

            regularization_loss=0
            loss_pre = loss_fn(predicts, class_labels)
            for item in range(len(masks)):
                regularization_loss += torch.norm(masks[item], 1)
            loss = loss_fucntion(inputs, outputs) + 0.000001 * regularization_loss + loss_pre
            loss_self = loss_l2(out_subnet_con[0],out_subnet[0]) + loss_l2(out_subnet_con[1],out_subnet[1]) + loss_l2(out_subnet_con[2],out_subnet[2])
            loss = loss + 0.00001*loss_self

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_re_list.append(loss.item())

            loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
            loop.set_postfix(loss=loss.item())
          
        print('epoch [{}/{}], loss_re:{:.4f}, '.format(epoch + 1, epochs, np.mean(loss_re_list)))
        write_loss_to_file(epoch + 1, epochs, np.mean(loss_re_list), 'Unified',model_name)

        if (epoch + 1) % 50 == 0:
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict(),
                        'subnet_w_fftmask':subnet_w_fftmask.state_dict(),
                        }, ckp_path + str(epoch+1) +'.pth' )



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    setup_seed(111)
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='mvtec')
    args = parser.parse_args()
    if args.dataset=='mvtec':
        model_name='DNP-KD_mvtec' # Please delete all 'license.txt' and 'readme.txt' files from the MVTec dataset.
    else:
        model_name='DNP-KD_visa'
    train(model_name)


