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
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.5, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/wgan/', help='base directory to save logs')
parser.add_argument('--data_root', default='data/', help='base directory to save logs')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=32, help='the height / width of the input image to network: 32 | 64')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='emotion_landscapes', help='dataset to train with: emotion_landscapes | cifar')
parser.add_argument('--z_dim', default=100, type=int, help='dimensionality of latent space')
parser.add_argument('--gp_lambda', type=int, default=10, help='gradient penalty hyperparam')
parser.add_argument('--critic_iters', type=int, default=5, help='gradient penalty hyperparam')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--nclass', type=int, default=9, help='number of classes (should be 9 for emotion landscapes)')
parser.add_argument('--save_model', action='store_true', help='if true, save the model throughout training')

opt = parser.parse_args()

name = 'z_dim=%d-lr=%.5f-gp_lambda=%d-critic_iters=%d' % (opt.z_dim, opt.lr, opt.gp_lambda, opt.critic_iters)
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
    netD = models.discriminator_64x64(opt.z_dim, opt.channels+opt.nclass, gan_type='wgan')
elif opt.image_width == 32:
    netG = models.generator_32x32(opt.z_dim+opt.nclass, opt.channels)
    netD = models.discriminator_32x32(opt.z_dim, opt.channels+opt.nclass, gan_type='wgan')
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
one = torch.FloatTensor([1]).cuda()
mone = one * -1

def plot_gen(epoch):
    nrow = opt.nclass 
    ncol = int(opt.batch_size/nrow) 

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

def calc_gradient_penalty(netD, real_data, fake_data):
    channels = real_data.shape[1]
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(opt.batch_size, int(real_data.nelement()/opt.batch_size)).contiguous().view(opt.batch_size, channels, 32, 32).cuda() 

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates.requires_grad = True

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.gp_lambda
    return gradient_penalty

def train_D(x):
    x, y = x

    # convert the integer y into a one_hot representation for D and G
    y_onehot = torch.Tensor(opt.batch_size, opt.nclass).cuda().zero_()
    y_onehot.scatter_(1, y.data.view(opt.batch_size, 1).long(), 1)
    y_D = y_onehot.view(opt.batch_size, opt.nclass, 1, 1).expand(opt.batch_size, opt.nclass, opt.image_width, opt.image_width)
    y_G = y_onehot.view(opt.batch_size, opt.nclass, 1, 1)

    # train discriminator
    netD.zero_grad()

    # real data
    out = netD(torch.cat([x, y_D], 1))
    D_real = out.mean()
    D_real.backward(mone)

    # fake data
    z = torch.randn(opt.batch_size, opt.z_dim, 1, 1).cuda()
    x_fake = netG(torch.cat([z, y_G], 1)) # generate from G
    out = netD(torch.cat([x_fake.detach(), y_D], 1)) # .detach() so we don't backprop through G (G is fixed while D trains)
    D_fake = out.mean()
    D_fake.backward(one)

    if opt.gp_lambda > 0:
        gradient_penalty = calc_gradient_penalty(netD, torch.cat([x.detach(), y_D], 1), torch.cat([x_fake.detach(), y_D], 1))
        gradient_penalty.backward()
    else:
        gradient_penalty = 0

    D_cost = D_fake - D_real + gradient_penalty
    wasserstein_D = D_real - D_fake
    optimizerD.step()

    return D_cost.item() #, wasserstein_D.item()

def train_G(x):
    x, y = x

    # convert the integer y into a one_hot representation for D and G
    y_onehot = torch.Tensor(opt.batch_size, opt.nclass).cuda().zero_()
    y_onehot.scatter_(1, y.data.view(opt.batch_size, 1).long(), 1)
    y_D = y_onehot.view(opt.batch_size, opt.nclass, 1, 1).expand(opt.batch_size, opt.nclass, opt.image_width, opt.image_width)
    y_G = y_onehot.view(opt.batch_size, opt.nclass, 1, 1)


    # train generator
    netG.zero_grad()

    z = torch.randn(opt.batch_size, opt.z_dim, 1, 1).cuda()
    x_fake = netG(torch.cat([z, y_G], 1)) # generate from G
    out = netD(torch.cat([x_fake, y_D], 1))
    G = out.mean()
    G.backward(mone)
    G_cost = -G
    optimizerG.step()

    return G_cost.item()

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    netD.train()
    netG.train()
    epoch_costD = 0
    epoch_costG = 0
    progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        progress.update(i+1)
        x = next(training_batch_generator)

        costD = 0
        for ii in range(opt.critic_iters):
            x = next(training_batch_generator)
            costD += train_D(x)
        costD /= opt.critic_iters

        x = next(training_batch_generator)
        costG = train_G(x)

        epoch_costD += costD
        epoch_costG += costG

    progress.finish()
    utils.clear_progressbar()

    print('[%02d] costD : %.5f | costG : %.5f (%d)' % (epoch, epoch_costD/opt.epoch_size, epoch_costG/opt.epoch_size, epoch*opt.epoch_size))

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
            
