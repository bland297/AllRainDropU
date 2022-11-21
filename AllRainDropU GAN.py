import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch_optimizer as optim
import torch.optim as optimizer
from Generator import Generator
from Discriminator import Discriminator
from collections import OrderedDict
import re
import CSI

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data_label = os.listdir(self.data_dir+'/label')
        lst_data_input = os.listdir(self.data_dir+'/input')

        lst_label = [f for f in lst_data_label]
        lst_input = [f for f in lst_data_input]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(self.data_dir + '/label/' + self.lst_label[index])
        input = np.load(self.data_dir + '/input/' + self.lst_input[index])

        # 정규화
        #label = label/255.0
        #input = input/255.0

        # 이미지와 레이블의 차원 = 2일 경우(채널이 없을 경우, 흑백 이미지), 새로운 채널(축) 생성
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        # transform이 정의되어 있다면 transform을 거친 데이터를 불러옴
        if self.transform:
            data = self.transform(data)

        return data

class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        # input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label.copy()), 'input': torch.from_numpy(input.copy())}

        return data


lr = 0.0003
batch_size = 12
num_epoch = 25

base_dir = 'C:/Users/--/AllRainDropUnetData'
data_dir = 'C:/Users/--/AllRainDropUnetData/data'
ckpt_dir = 'C:/Users/--/AllRainDropUnetData/checkpoint/'
log_dir = 'C:/Users/--/AllRainDropUnetData/log'
result_dir = 'C:/Users/--/AllRainDropUnetData/result'
load_model_epoch = 'model_epoch15.pth'


transform = transforms.Compose([ToTensor()])
dataset_train = Dataset(data_dir=data_dir+'/train', transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
# dataset_val = Dataset(data_dir=data_dir+'/validation', transform=transform)
# loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

# 네트워크 생성하기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Generator().to(device)
net.apply(weights_init)

## 네트워크 저장하기
def save(ckpt_dir, gen, dis, gop, dop, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'generator': gen.state_dict(), 'discriminator': dis.state_dict(),
                'goptim':gop.state_dict(), 'doptim':dop.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir,name):
    # ckpt_lst = os.listdir(ckpt_dir)
    # ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir,name))
    gen = Generator().to(device)
    new_state_dict = OrderedDict()
    for k, v in dict_model['generator'].items():
        aname = k.replace("model.","")
        new_state_dict[aname] = v
    gen.load_state_dict(new_state_dict)


    dis = Discriminator().to(device)
    new_state_dict = OrderedDict()
    for k, v in dict_model['discriminator'].items():
        aname = k.replace("model.","")
        new_state_dict[aname] = v
    dis.load_state_dict(new_state_dict)

    gen_params = gen.parameters()
    gop = optim.Yogi(gen_params, lr=0.00003, betas=(0.5,0.999),eps=1e-3,initial_accumulator=1e-6,weight_decay=0)
    gop.load_state_dict(dict_model['goptim'])
    dis_params = dis.parameters()
    dop = optim.Yogi(dis_params, lr=0.0001, betas=(0.9,0.999),eps=1e-3,initial_accumulator=1e-6,weight_decay=0)
    dop.load_state_dict(dict_model['doptim'])
    epoch = int(re.sub(r'[^0-9]', '', name))

    return gen, dis, gop, dop, epoch
    return gen, gop, epoch

# 손실함수 정의하기
#fn_loss = nn.BCEWithLogitsLoss().to(device)

net_params = net.parameters()

# Optimizer 설정하기6
# g_optim = optim.Yogi(net_params, lr=0.00005, betas=(0.5,0.999),eps=1e-3,initial_accumulator=1e-6,weight_decay=0)
g_optim = optim.Yogi(net_params, lr=0.00005, betas=(0.5,0.999),eps=1e-3,initial_accumulator=1e-6,weight_decay=0)
# g_optim = optimizer.Adam(net_params, lr=0.00005, betas=(0.0,0.999))
g_scheduler = optimizer.lr_scheduler.LambdaLR(optimizer=g_optim,
                        lr_lambda=lambda epoch:(0.98**epoch),
                        last_epoch=-1,
                        verbose=False)
 
discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
dis_params = discriminator.parameters()
d_optim = optim.Yogi(dis_params, lr=0.0002, betas=(0.9,0.999),eps=1e-3,initial_accumulator=1e-6,weight_decay=0)
# d_optim = optimizer.Adam(dis_params, lr=0.0002, betas=(0.5,0.999))
d_scheduler = optimizer.lr_scheduler.LambdaLR(optimizer=d_optim,
                        lr_lambda=lambda epoch:(0.98**epoch),
                        last_epoch=-1,
                        verbose=False)

criterion = nn.BCELoss()


num_data_train = len(dataset_train)
# num_data_val = len(dataset_val) 
num_batch_train = np.ceil(num_data_train / batch_size)
# num_batch_val = np.ceil(num_data_val / batch_size)

# # Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
# writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

# 네트워크 학습시키기
st_epoch = 0
# 학습한 모델이 있을 경우 모델 로드하기
# net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim) 
net, discriminator, g_optim, d_optim, st_epoch = load(ckpt_dir=ckpt_dir,name=load_model_epoch)
            
def train():
    for epoch in range(st_epoch + 1, num_epoch + 1):
            net.train()
            discriminator.train()

            gloss_arr = []
            dloss_arr = []
            for batch, data in enumerate(loader_train, 1):



                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)
                # print(input[:,-1].shape,label.shape)

                # real_label = torch.full((label.shape[0], 1), 1, dtype=torch.float32).to(device)
                # fake_label = torch.full((label.shape[0], 1), 0, dtype=torch.float32).to(device)

                # backward pass
                g_optim.zero_grad()
                d_optim.zero_grad()
                
                # z = torch.randn(label.shape[0], 100,1,1).to(device)
                # output = net(input,z)
                output = net(input)

                pred_real = discriminator(label)
                # real_loss = criterion(pred_real,real_label)
                real_loss = torch.mean((pred_real-1)**2)/2
                real_loss.backward()

                pred_fake = discriminator(output.detach())
                # fake_loss = criterion(disc,fake_label)
                fake_loss = torch.mean((pred_fake)**2)/2
                fake_loss.backward()
                
                d_loss = (fake_loss + real_loss)/2
                # d_loss.backward()
                d_optim.step()

                net.zero_grad()
                disc = discriminator(output)
                # g_loss = criterion(disc,real_label)
                g_loss = torch.mean((disc-1)**2)
                g_loss += torch.mean(torch.log(torch.cosh(output-label)))
                # g_loss=g_loss1+g_loss2
                g_loss.backward()
                g_optim.step()

                # 손실함수 계산 
                gloss_arr += [g_loss.item()]
                dloss_arr += [d_loss.item()]

                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSSg %.4f | LOSSd %.4f" %
                    (epoch, num_epoch, batch, num_batch_train, np.mean(gloss_arr), np.mean(dloss_arr)))

                if(batch%100==0):
                    with torch.no_grad():
                        net.eval()
                        output = net(input)
                        plt.imsave(os.path.join(result_dir, 'trainpng', '%d_%d_TE_output.png' % (epoch,batch)), (output[0].cpu()).detach().numpy().squeeze(),
                            cmap='jet',vmin=-2.5,vmax=2.5)
                        plt.imsave(os.path.join(result_dir, 'trainpng', '%d_%d_TE_label.png' % (epoch,batch)), label[0].cpu().squeeze(),
                            cmap='jet',vmin=-2.5,vmax=2.5)
                        net.train()

            writer_train.add_scalar('gloss', np.mean(gloss_arr), epoch)
            writer_train.add_scalar('dloss', np.mean(dloss_arr), epoch)

            # with torch.no_grad():
            #     net.eval()
            #     loss_arr = []

            #     for batch, data in enumerate(loader_val, 1):
            #         # forward pass
            #         label = data['label'].to(device)
            #         input = data['input'].to(device) 

            #         output = net(input)

            #         # 손실함수 계산하기
            #         #loss = fn_loss(output, label)
            #         loss = torch.mean(torch.log(torch.cosh(output-label)))

            #         loss_arr += [loss.item()]

            #         print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
            #             (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

            # writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

            g_scheduler.step()
            d_scheduler.step() 

            # epoch 50마다 모델 저장하기
            if epoch % 5 == 0:
                # torch.save()
                save(ckpt_dir=ckpt_dir, gen=net, dis=discriminator, gop=g_optim, dop=d_optim, epoch=epoch)
            writer_train.close()
            # writer_val.close()

def test():
    transform = transforms.Compose([ToTensor()])

    dataset_test = Dataset(data_dir=data_dir+'/test', transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

    # 결과 디렉토리 생성하기
    result_dir = os.path.join(base_dir, 'result')
    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir, 'png'))
        os.makedirs(os.path.join(result_dir, 'numpy'))


    net, discriminator, g_optim, d_optim, st_epoch = load(ckpt_dir=ckpt_dir,name=load_model_epoch)
    # net, g_optim,st_epoch = load(ckpt_dir=ckpt_dir,name=load_model_epoch)
    net.to(device)
    # discriminator.to(device)
    with torch.no_grad():
        net.eval()
        #discriminator.eval()

        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input).detach()

            # 손실함수 계산하기
            #loss = fn_loss(output, label)
            loss = torch.mean(torch.log(torch.cosh(output-label)))

            loss_arr += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                    (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard 저장하기
            # label = fn_tonumpy(label)
            # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))  
            # output = fn_tonumpy(fn_class(output))

            # 테스트 결과 저장하기
            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'png', '%04d_label.png' % id), label[j].cpu().squeeze(), cmap='jet',vmin=-2.5,vmax=2.5)
                plt.imsave(os.path.join(result_dir, 'png', '%04d_output.png' % id), output[j].cpu().squeeze(), cmap='jet',vmin=-2.5,vmax=2.5)

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].cpu().squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].cpu().squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].cpu().squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
            (batch, num_batch_test, np.mean(loss_arr)))
        

if __name__ == '__main__':
    
    train()

    test()

    CSI.print_csimean()
