import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import utils
import itertools
import progressbar
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.5, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/gan/', help='base directory to save logs')
parser.add_argument('--data_root', default='data/', help='base directory to save logs')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=32, help='the height / width of the input image to network: 32 | 64')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='emotion_landscapes', help='dataset to train with: emotion_landscapes | cifar')
parser.add_argument('--z_dim', default=64, type=int, help='dimensionality of latent space')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--nclass', type=int, default=9, help='number of classes (should be 9 for emotion landscapes)')
parser.add_argument('--save_model', action='store_true', help='if true, save the model throughout training')

opt = parser.parse_args()

name = 'z_dim=%d-lr=%.5f' % (opt.z_dim, opt.lr)
opt.log_dir = '%s/%s_%dx%d/%s' % (opt.log_dir, opt.dataset, opt.image_width, opt.image_width, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)

# ---------------- load the models ----------------
# the generator maps [z, y_vec] -> x
# z dimensions: [batch_size, z_dim, 1, 1]
# y_vec dimensions: [batch_size, nclass, 1, 1]
#
# the discriminator maps [x, y_im] -> prediction
# x dimensions: [batch_size, channels, image_width, image_width]
# y_im dimensions: [batch_size, nclass, image_width, image_width] (i.e., one hot vector expanded to be of dimensionality of image)
import models.dcgan as models
if opt.image_width == 64:
    netG = models.generator_64x64(opt.z_dim+opt.nclass, opt.channels)
    netD = models.discriminator_64x64(opt.z_dim, opt.channels+opt.nclass)
elif opt.image_width == 32:
    netG = models.generator_32x32(opt.z_dim+opt.nclass, opt.channels)
    netD = models.discriminator_32x32(opt.z_dim, opt.channels+opt.nclass)
else:
    raise ValueError('Invalid image width %d' % opt.image_width)


netD.apply(utils.init_weights)
netG.apply(utils.init_weights)

optimizerD = opt.optimizer(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = opt.optimizer(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

netD.cuda()
netG.cuda()

# loss function for discriminator
criterion = nn.BCELoss()
criterion.cuda()

# ---------------- datasets ----------------
trainset = utils.load_dataset(opt) 
train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=opt.data_threads)


def get_training_batch():
    while True:
        for x, y in train_loader:
            yield [x.cuda(), y.cuda()]
training_batch_generator = get_training_batch()

# so all our generations use same noise vector - useful for visualizaiton purposes
z_fixed = torch.randn(opt.batch_size, opt.z_dim, 1, 1).cuda()
real_label = 1
fake_label = 0

def plot_gen(epoch):
    nrow = opt.nclass 
    ncol = 10 

    # randomly sample classes
    y_onehot = torch.Tensor(opt.batch_size, opt.nclass, 1, 1).cuda().zero_()
    for i in range(nrow):
        for j in range(ncol):
            y_onehot[i*ncol+j][i] = 1
    gen = netG(torch.cat([z_fixed, y_onehot], 1))

    to_plot = []
    for i in range(nrow):
        row = []
        for j in range(ncol):
            row.append(gen[i*ncol+j])
        to_plot.append(row)

    fname = '%s/gen/%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)

def train(x):
    x, y = x

    # convert the integer y into a one_hot representation for D and G
    y_onehot = torch.Tensor(opt.batch_size, opt.nclass).cuda().zero_()
    y_onehot.scatter_(1, y.data.view(opt.batch_size, 1).long(), 1)
    y_D = y_onehot.view(opt.batch_size, opt.nclass, 1, 1).expand(opt.batch_size, opt.nclass, opt.image_width, opt.image_width)
    y_G = y_onehot.view(opt.batch_size, opt.nclass, 1, 1)

    label = torch.Tensor(opt.batch_size,).cuda()
    # train discriminator
    netD.zero_grad()

    # real data
    label.fill_(real_label)
    out = netD(torch.cat([x, y_D], 1))
    errD_real = criterion(out, label)
    errD_real.backward()
    acc_real = errD_real.gt(0.5).sum()

    # fake data
    label.fill_(fake_label)
    z = torch.randn(opt.batch_size, opt.z_dim, 1, 1).cuda()
    x_fake = netG(torch.cat([z, y_G], 1)) # generate from G
    out = netD(torch.cat([x_fake.detach(), y_D], 1)) # .detach() so we don't backprop through G (G is fixed while D trains)
    errD_fake = criterion(out, label)
    errD_fake.backward()
    acc_fake = errD_fake.lt(0.5).sum()

    errD = errD_real +errD_fake
    optimizerD.step()

    # train generator
    netG.zero_grad()
    label.fill_(real_label)
    out = netD(torch.cat([x_fake, y_D], 1))
    errG = criterion(out, label)
    errG.backward()
    optimizerG.step()

    return errD.item(), errG.item(), acc_real.item(), acc_fake.item()

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    netD.train()
    netG.train()
    epoch_errD = 0
    epoch_errG = 0
    epoch_acc_real = 0
    epoch_acc_fake = 0
    progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        progress.update(i+1)
        x = next(training_batch_generator)

        errD, errG, acc_real, acc_fake = train(x)
        epoch_errD += errD
        epoch_errG += errG
        epoch_acc_real += acc_real
        epoch_acc_fake += acc_fake

    progress.finish()
    utils.clear_progressbar()

    print('[%02d] errD : %.5f | errG : %.5f | acc real : %.5f | acc fake : %.5f (%d)' % (epoch, epoch_errD/opt.epoch_size, epoch_errG/opt.epoch_size, epoch_acc_real/opt.epoch_size, epoch_acc_fake/opt.epoch_size, epoch*opt.epoch_size))

    # plot some stuff
    #netG.eval()
    plot_gen(epoch)

    # save the model
    if opt.save_model and epoch % 10 == 0:
        torch.save({
            'netD': netD,
            'netG': netG,
            'opt': opt},
            '%s/model.pth' % opt.log_dir)
    if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)
            
