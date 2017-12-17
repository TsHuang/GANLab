from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import h5py
import numpy as np
from torch.autograd import Variable
import math


class CelebADataset(object):
    def __init__(self, h5_path, transform=None):
        assert (os.path.isfile(h5_path))
        self.h5_path = h5_path
        self.transform = transform

        # loading the dataset into memory
        f = h5py.File(self.h5_path, "r")
        key = list(f.keys())
        print("key list:", key)
        self.dataset = f[key[0]]
        print("dataset loaded and its shape:", self.dataset.shape)

    def __getitem__(self, index):
        img = self.dataset[index]
        img = np.transpose(img, (1, 2, 0))
        if self.transform is not None:
            img = self.transform(img)

        return img, 0

    def __len__(self):
        return len(self.dataset)


parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
# parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=2048, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--saveInt', type=int, default=1, help='number of epochs between checkpoints')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate, default=0.0003')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./results1218', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

# save training loss
curve_file = opt.outf + '/curveData.csv'
report = np.array(['Epoch', 'dataIdx', 'Loss_VAE', 'Loss_D', 'Loss_G', 'D(x)', 'D(G(z))_up', 'D(G(z))_down'])

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if not os.path.isdir(opt.outf):
    os.mkdir(opt.outf)

# if opt.dataset in ['imagenet', 'folder', 'lfw']:
#     # folder dataset
#     dataset = dset.ImageFolder(root=opt.dataroot,
#                                transform=transforms.Compose([
#                                    transforms.Scale(opt.imageSize),
#                                    transforms.CenterCrop(opt.imageSize),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                ]))
# elif opt.dataset == 'lsun':
#     dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
#                         transform=transforms.Compose([
#                             transforms.Scale(opt.imageSize),
#                             transforms.CenterCrop(opt.imageSize),
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                         ]))
# elif opt.dataset == 'cifar10':
#     dataset = dset.CIFAR10(root=opt.dataroot, download=True,
#                            transform=transforms.Compose([
#                                transforms.Scale(opt.imageSize),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))
# elif opt.dataset == 'fake':
#     dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
#                             transform=transforms.ToTensor())
# assert dataset
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

#batch_size = 64

T = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
dataloader = torch.utils.data.DataLoader(CelebADataset('../CelebA_aligned.h5', transform=T), batch_size=opt.batchSize,
                                         shuffle=True, num_workers=1)

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _Sampler(nn.Module):
    def __init__(self):
        super(_Sampler, self).__init__()

    def forward(self, input):
        mu = input[0]
        logvar = input[1]

        std = logvar.mul(0.5).exp_()  # calculate the STDEV
        if opt.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()  # random normalized noise
        else:
            eps = torch.FloatTensor(std.size()).normal_()  # random normalized noise
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


class _Encoder(nn.Module):
    def __init__(self):
        super(_Encoder, self).__init__()

        self.conv1 = nn.Conv2d(nc, ngf, 5, 2, 2, bias=False)  # 64->32
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 5, 2, 2, bias=False)  # 32->16
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 5, 2, 2, bias=False)  # 16->8

        self.fc1 = nn.Linear(ngf * 4 * 8 * 8, nz)
        self.bn1 = nn.BatchNorm1d(nz)
        self.mean = nn.Linear(nz, nz)

        self.fc2 = nn.Linear(ngf * 4 * 8 * 8, nz)
        self.bn2 = nn.BatchNorm1d(nz)
        self.relu = nn.ReLU(True)
        self.log_var = nn.Linear(nz, nz)

    def forward(self, input):
        x = self.conv3(self.conv2(self.conv1(input)))

        x = x.view(-1, ngf * 4 * 8 * 8)
        x1 = self.mean(self.bn1(self.fc1(x)))  # mean
        x2 = self.log_var(self.bn2(self.fc2(x)))  # log_var

        return [x1, x2]


class _Decoder(nn.Module):
    def __init__(self):
        super(_Decoder, self).__init__()

        self.fc1 = nn.Linear(nz, ngf * 4 * 8 * 8)
        self.bn1 = nn.BatchNorm1d(ngf * 4 * 8 * 8)
        self.relu = nn.ReLU(True)
        self.deconv1 = nn.ConvTranspose2d(ngf * 4, ngf * 4, 5, 2, 2, 1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 2, 2, 1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(ngf * 2, int(ngf * 0.5), 5, 2, 2, 1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(int(ngf * 0.5), nc, 5, 1, 2, 0, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, input):
        x = input.view(-1, nz)
        x = self.fc1(x)
        x = self.relu(self.bn1(x))
        x = x.view(-1, ngf * 4, 8, 8)
        x = self.deconv4(self.deconv3(self.deconv2(self.deconv1(x))))

        output = self.tanh(x)
        return output


class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()

        self.encoder = _Encoder()
        self.sampler = _Sampler()
        self.decoder = _Decoder()

    def forward(self, input):
        output = self.decoder(self.sampler(self.encoder(input)))

        return output

    def make_cuda(self):
        self.encoder.cuda()
        self.sampler.cuda()
        self.decoder.cuda()


# netG = _netG(opt.imageSize, ngpu)
netG = _netG()
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))  # load pretrained
print(netG)


class _netD(nn.Module):  # Discriminator
    def __init__(self):
        super(_netD, self).__init__()

        self.conv1 = nn.Conv2d(nc, ndf, 5, 1, 2, bias=False)
        self.bn2d_1 = nn.BatchNorm2d(ndf)
        self.conv2 = nn.Conv2d(ndf, 4 * ndf, 5, 2, 2, bias=False)
        self.bn2d_2 = nn.BatchNorm2d(4 * ndf)
        self.conv3 = nn.Conv2d(4 * ndf, 8 * ndf, 5, 2, 2, bias=False)
        self.bn2d_3 = nn.BatchNorm2d(8 * ndf)
        self.conv4 = nn.Conv2d(8 * ndf, 8 * ndf, 5, 2, 2, bias=False)
        self.bn2d_4 = nn.BatchNorm2d(8 * ndf)
        self.fc1 = nn.Linear(8 * ndf * 8 * 8, 1024)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        self.relu3 = nn.ReLU(True)
        self.relu4 = nn.ReLU(True)
        self.elu = nn.ELU(1.0, True)
        self.fc2 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        x = self.relu4(self.bn2d_4(self.conv4(self.relu3(
            self.bn2d_3(self.conv3(self.relu2(self.bn2d_2(self.conv2(self.relu1(self.bn2d_1(self.conv1(input))))))))))))
        x = x.view(-1, 8 * ndf * 8 * 8)
        x = self.fc1(x)
        e = self.elu(x)
        x = self.fc2(e)
        output = self.sig(x)

        return output.view(-1, 1).squeeze(1), e


# netD = _netD(opt.imageSize, ngpu)
netD = _netD()
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()
MSECriterion = nn.MSELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label1 = torch.FloatTensor(opt.batchSize)
label0 = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.make_cuda()
    criterion.cuda()
    MSECriterion.cuda()
    input, label1, label0 = input.cuda(), label1.cuda(), label0.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
label1 = Variable(label1)
label0 = Variable(label0)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

beta = 15
gamma = 5
alpha = 0.1

# setup optimizer
# optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.RMSprop(netD.parameters(), lr=alpha * opt.lr)
optimizerG_en = optim.RMSprop(netG.encoder.parameters(), lr=opt.lr)
optimizerG_de = optim.RMSprop(netG.decoder.parameters(), lr=opt.lr)

gen_win = None
rec_win = None

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(decoder(z))) + log(1 - D(decoder(encoder(x))))
        ###########################
        # train with real
        # netD.zero_grad()

        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        # input.resize_as_(real_cpu).copy_(real_cpu)


        encoded = netG.encoder(input)
        sampled = netG.sampler(encoded)
        rec = netG.decoder(sampled)
        mu = encoded[0]
        logvar = encoded[1]

        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)
        rec_noise = netG.decoder(noise)

        # -------------------
        label1.data.resize_(real_cpu.size(0)).fill_(real_label)
        # train with real D(x)
        output, _ = netD(input)
        errD_real = criterion(output, label1)
        # errD_real.backward()
        D_x = output.data.mean()

        label0.data.resize_(real_cpu.size(0)).fill_(fake_label)

        # train with fake D(decoder(encoder(input)))
        output, _ = netD(rec)
        errD_fake2 = criterion(output, label0)
        D_G_z11 = output.data.mean()

        # train with fake D(decoder(z))
        output, _ = netD(rec_noise)
        errD_fake1 = criterion(output, label0)

        D_G_z1 = output.data.mean()

        # discriminator loss
        errD = errD_real + errD_fake1 + errD_fake2
        netD.zero_grad()
        errD.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G. decoder network: maximize log(D(decoder(z))) + log(D(decoder(encoder(x)))) - reconstruction loss
        ###########################

        output, e_rec = netD(rec)
        D_G_z2 = output.data.mean()
        errG1 = criterion(output, label1)

        # reconstruction loss: MSE in feature domain
        output, e_input = netD(input)
        MSEerr = MSECriterion(e_rec, e_input.detach())  # enhance by gamma

        # log(D(decoder(z)))
        output, _ = netD(rec_noise)
        D_G_z22 = output.data.mean()
        errG2 = criterion(output, label1)

        # decoder loss
        errG = errG1 + errG2 + gamma * MSEerr
        netG.decoder.zero_grad()
        errG.backward(retain_graph=True)
        optimizerG_de.step()

        ############################
        # (3) Update G.encoder network: minimize prior loss + reconstruction loss
        ###########################

        # prior loss: KLD
        prior_loss = 1 + logvar - mu.pow(2) - logvar.exp()
        KLD = (-0.5 * torch.sum(prior_loss)) / torch.numel(mu.data)

        # reconstruction loss: MSE in feature domain

        # encoder loss
        VAEerr = beta * KLD + MSEerr
        netG.encoder.zero_grad()
        VAEerr.backward()
        optimizerG_en.step()

        print('[%d/%d][%d/%d] Loss_VAE: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 VAEerr.data[0], errD.data[0], errD.data[0], D_x, D_G_z1 + D_G_z11, D_G_z2 + D_G_z22))

        newdata = np.array(
            [epoch, i, VAEerr.data[0], errD.data[0], errD.data[0], D_x, D_G_z1 + D_G_z11, D_G_z2 + D_G_z22])

        report = np.vstack((report, newdata))

        if i % 100 == 0:
            vutils.save_image(real_cpu,
                              '%s/real_samples.png' % opt.outf,
                              normalize=True)

            fake1 = netG.decoder(sampled)
            vutils.save_image(fake1.data,
                              '%s/fake_rec_samples_epoch_%03d.png' % (opt.outf, epoch),
                              normalize=True)

            fake2 = netG.decoder(fixed_noise)
            vutils.save_image(fake2.data,
                              '%s/fake_noise_samples_epoch_%03d.png' % (opt.outf, epoch),
                              normalize=True)

    # do checkpointing
    if epoch % opt.saveInt == 0 and epoch != 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

# save curve data
np.savetxt(curve_file, report, fmt="%s", delimiter=",")
