import os
import cv2
import math
import copy
import random
import argparse
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial import distance
from skimage.feature import canny
from sklearn.metrics import accuracy_score
from skimage.measure import compare_ssim, compare_psnr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.loss import GANLoss, AdversarialLoss, PerceptualLoss, StyleLoss
from utils.RTV import RTV

def list_from_file(filename, prefix='', offset=0, max_num=0, encoding='utf-8'):
    cnt = 0
    item_list = []
    with open(filename, 'r', encoding=encoding) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip('\n\r'))
            cnt += 1
    return item_list

gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device = 'cuda:'+gpu_id if torch.cuda.is_available() else 'cpu'
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.03)



class SIFTDataset(Dataset):
    def __init__(self, root, size=(256,1024), mode="train"):
        if mode=="train":
            file_list = os.path.join(root, "train.txt")
        else:
            file_list = os.path.join(root, "val.txt")
        self.img_dir = os.path.join(root, "image_2")
        self.lp_dir = os.path.join(root, "LTP")
        self.size = size
        self.filelist = list_from_file(file_list)

    def __getitem__(self, idx):
        return self.load_item(idx)

    def __len__(self):
        return self.len(self.filelist)

    def load_item(self, idx):
        filename = f'{self.filelist[idx]}.png'
        H, W = self.size
        Ig = cv2.imread(os.path.join(self.img_dir, filename))
        Ig = cv2.resize(Ig, (H, W))
        Si = self.load_sift(Ig)
        Lg = self.load_lp(filename)
        Lg = Lg.astype('float') / 127.5 - 1.
        Lg = np.reshape(Lg, (H, W, 1))
        rtn = Lg
        Ig = Ig.astype('float') / 127.5 - 1.
        return self.tensor(Ig), self.tensor(Si), self.tensor(rtn), filename

    def load_sift(self, img):
        H, W = self.size
        fealen = 128
        feature = np.zeros([H, W, fealen], dtype=float)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        meta, des = sift.detectAndCompute(np.uint8(img), None)
        if len(meta) == 0:
            return feature
        des = des.astype('float') / 127.5 - 1.
        used = []
        for i in range(len(meta)):
            a = int(math.ceil(meta[i].pt[1]) - 1)
            b = int(math.ceil(meta[i].pt[0])) - 1
            fea = list(des[i])
            if self.isEmpty(feature[a][b][:128]):
                feature[a][b][:128] = fea
                used.append(i)
        if True:  # Reduce collisions, can be skip
            meta = np.delete(meta, used)
            for i in range(len(meta)):
                a = int(math.ceil(meta[i].pt[1]) - 1)
                b = int(math.ceil(meta[i].pt[0])) - 1
                ra, rb = self.search_ab(feature, a, b, H, W)
                if ra == -1:
                    continue
                feature[ra][rb][:128] = list(des[i])
        return feature


    def load_lp(self, name):
        lp_img = cv2.imread(os.path.join(self.lp_dir, name))
        lp_img = cv2.cvtColor(lp_img, cv2.COLOR_BGR2GRAY)

        return lp_img


    def isEmpty(self, feature):
        for i in range(min(len(feature), 128)):
            if feature[i] != 0:
                return False
        return True

    def search_ab(self, feature, a, b, h, w):
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                ra = a + i
                rb = b + j
                if 0 <= ra <= h - 1 and 0 <= rb <= w - 1 and self.isEmpty(feature[ra][rb]):
                    return ra, rb
        return -1, -1

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(init_func)


class LBPGenerator(BaseNetwork):
    def __init__(self, in_channels=128, out_channels=1):
        super(LBPGenerator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
        )
        self.layer3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
        )
        self.layer4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
        )
        self.layer5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
        )
        self.layer6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
        )
        self.layer7 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512)
        )
        # It's feasible to use UpsamplingNearest2d + Conv2d or ConvTranspose2d directly.
        self.layer8 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512)
        )
        self.layer9 = nn.Sequential(
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512)
        )
        self.layer10 = nn.Sequential(
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512)
        )
        self.layer11 = nn.Sequential(
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512)
        )
        self.layer12 = nn.Sequential(
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
        )
        self.layer13 = nn.Sequential(
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
        )
        self.layer14 = nn.Sequential(
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64)
        )
        self.layer15 = nn.Sequential(
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.init_weights()

    def forward(self, Si):
        layer1 = self.layer1(Si)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        layer8 = self.layer8(layer7)
        layer9 = self.layer9(torch.cat([layer8, layer7], dim=1))
        layer10 = self.layer10(torch.cat([layer9, layer6], dim=1))
        layer11 = self.layer11(torch.cat([layer10, layer5], dim=1))
        layer12 = self.layer12(torch.cat([layer11, layer4], dim=1))
        layer13 = self.layer13(torch.cat([layer12, layer3], dim=1))
        layer14 = self.layer14(torch.cat([layer13, layer2], dim=1))
        layer15 = self.layer15(torch.cat([layer14, layer1], dim=1))
        Lo = torch.tanh(layer15)
        return Lo

class Discriminator(BaseNetwork):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.use_sigmoid = True
        self.conv1 = self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=True),
        )

        self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.save_dir = 'weights/'


class LBPModel(BaseModel):
    def __init__(self):
        super(LBPModel, self).__init__()
        self.lr = 1e-4
        self.gan_type = 're_avg_gan'
        self.channels = 3
        self.gen = nn.DataParallel(LBPGenerator(in_channels=128, out_channels=self.channels)).cuda()
        self.dis = nn.DataParallel(Discriminator(in_channels=self.channels)).cuda()
        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.criterionGAN = GANLoss(gan_type=self.gan_type)

        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.dis_optimizer = optim.Adam(self.dis.parameters(), lr=self.lr, betas=(0.9, 0.999))

        self.ADV_LOSS_WEIGHT = 0.2
        self.L1_LOSS_WEIGHT = 100
        self.PERC_LOSS_WEIGHT = 1

    def process(self, Si, Lg):
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        Lo = self(Si)

        if self.gan_type == 're_avg_gan':
            dis_fake, _ = self.dis(Lo.detach())
            gen_fake, _ = self.dis(Lo)
            dis_real, _ = self.dis(Lg)
            gen_real, _ = self.dis(Lg)
            dis_loss = self.criterionGAN(dis_real - dis_fake, True)
            gen_gan_loss = (self.criterionGAN(gen_real - torch.mean(gen_fake), False) +
                            self.criterionGAN(gen_fake - torch.mean(gen_real), True)) / 2. * self.ADV_LOSS_WEIGHT
        else:
            dis_input_real = Lg
            dis_input_fake = Lo.detach()
            dis_real, _ = self.dis(dis_input_real)
            dis_fake, _ = self.dis(dis_input_fake)
            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss = (dis_real_loss + dis_fake_loss) / 2

            gen_input_fake = Lo
            gen_fake, _ = self.dis(gen_input_fake)
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.ADV_LOSS_WEIGHT
        if self.channels == 1:
            gen_perceptual_loss = self.perceptual_loss(torch.cat([Lo, Lo, Lo], dim=1), torch.cat([Lg, Lg, Lg], dim=1)) * self.PERC_LOSS_WEIGHT
        else:
            gen_perceptual_loss = self.perceptual_loss(Lo, Lg) * self.PERC_LOSS_WEIGHT
        gen_l1_loss = self.l1_loss(Lo, Lg) * self.L1_LOSS_WEIGHT
        gen_loss = gen_gan_loss + gen_perceptual_loss + gen_l1_loss
        return Lo, gen_loss, dis_loss

    def forward(self, Si):
        return self.gen(Si)

    def backward(self, gen_loss=None, dis_loss=None, retain_graph=False):
        if dis_loss is not None:
            dis_loss.backward(retain_graph=retain_graph)
            self.dis_optimizer.step()

        if gen_loss is not None:
            gen_loss.backward(retain_graph=retain_graph)
            self.gen_optimizer.step()

    def save(self, path, epoch):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.gen.state_dict(), os.path.join(self.save_dir, path, 'ltp_gen_{}.pth'.format(epoch)))
        torch.save(self.dis.state_dict(), os.path.join(self.save_dir, path, 'ltp_dis_{}.pth'.format(epoch)))

    def load(self, path):
        self.gen.load_state_dict(torch.load(os.path.join(self.save_dir, path)))


class SIFTReconstruction():
    def __init__(self, root="data", size=(256, 1024), batch_size=1, num_epoches=50,num_checkpoint=5):
        self.dataset = 'kitti'
        self.batch_size = batch_size

        train_dataset = SIFTDataset(root, size,mode="train")
        test_dataset = SIFTDataset(root, size,mode="val")

        self.lbp_model = LBPModel().cuda()
        self.n_epochs = num_epoches
        self.num_checkpoint= num_checkpoint
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)



    def train(self):
        print('\nTrain/Val: ')
        for epoch in range(self.n_epochs):
            gen_losses, ssim, psnr = [], [], []
            for cnt, items in enumerate(self.train_loader):
                self.lbp_model.train()

                Ig, Si, Lg = (item.cuda() for item in items[:-1])

                Lo, gen_lbp_loss, dis_lbp_loss = self.lbp_model.process(Si, Lg)
                self.lbp_model.backward(gen_lbp_loss, dis_lbp_loss)
                s, p = self.metrics(Lg, Lo)
                ssim.append(s)
                psnr.append(p)

                gen_losses.append(gen_lbp_loss.item())

                print('Train (%d) Loss:%5.4f, SSIM:%4.4f, PSNR:%4.2f' %
                      (cnt, np.mean(gen_losses), np.mean(ssim), np.mean(psnr)), end='\r')
                if epoch % self.num_checkpoint == 0:
                    val_ssim, val_psnr = self.test()
                    self.lbp_model.save('kitti/', epoch)
                    print('Val (%d) SSIM:%4.4f, PSNR:%4.2f' % (cnt, val_ssim, val_psnr))

    def test(self, pretrained=None):
        if pretrained:
            print('\nTest:')
            self.lbp_model.load(pretrained)

        self.lbp_model.eval()

        ssim, psnr = [], []
        for cnt, items in enumerate(self.test_loader):
            Ig, Si, Lg = (item.cuda() for item in items[:-1])

            Lo = self.lbp_model(Si)


            s, p = self.metrics(Lg, Lo)
            ssim.append(s)
            psnr.append(p)
            if cnt < 100:
                Lo = self.postprocess(Lo)
                cv2.imwrite('results/Lo_%06d.jpg' % (cnt+1), Lo[0])
                # Io = self.postprocess(Io)
                # cv2.imwrite('res/' + self.choice + '_results/Io_%06d.jpg' % (cnt+1), Io[0])
        if pretrained:
            print(' Evaluation: SSIM:%4.4f, PSNR:%4.2f' % (np.mean(ssim), np.mean(psnr)))
        return np.mean(ssim), np.mean(psnr)

    def postprocess(self, img):
        img = img * 127.5 + 127.5
        img = img.permute(0, 2, 3, 1)
        return img.int().cpu().detach().numpy()

    def metrics(self, Ig, Io):
        a = self.postprocess(Ig)
        b = self.postprocess(Io)
        ssim, psnr = [], []
        for i in range(len(a)):
            ssim.append(compare_ssim(a[i], b[i], win_size=11, data_range=255.0, multichannel=True))
            psnr.append(compare_psnr(a[i], b[i], data_range=255))
        return np.mean(ssim), np.mean(psnr)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, required=True, help='train or test the model')
    parser.add_argument('--num_epoches', '-n', type=int, default=50, help="Number of epoches")
    parser.add_argument('--num_checkpoint', '-s', type=int, default=5, help="Number of epoches")
    parser.add_argument("--b_size", "-b", type=int, default=1, help="Batch Size")
    parser.add_argument("--pretrained", "-p", type=str, default=None, help="Pretrained dir for test only")

    args = parser.parse_args()

    model = SIFTReconstruction(root="data", size=(256, 1024), batch_size=args.b_size, num_epoches=args.num_epoches,num_checkpoint=args.num_checkpoint)
    if args.mode == 'train':
        model.train()
    elif args.mode == 'test':
        model.test(args.pretrained)
    print('End.')
