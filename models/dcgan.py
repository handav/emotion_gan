import torch
import torch.nn as nn

from torch.autograd import Variable

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class discriminator_64x64(nn.Module):
    def __init__(self, z_dim, nc=1, nf=64, gan_type='gan'):
        super(discriminator_64x64, self).__init__()
        self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                dcgan_conv(nc, nf),
                # state size. (nf) x 32 x 32
                dcgan_conv(nf, nf * 2),
                # state size. (nf*2) x 16 x 16
                dcgan_conv(nf * 2, nf * 4),
                # state size. (nf*4) x 8 x 8
                dcgan_conv(nf * 4, nf * 8),
                # state size. (nf*8) x 4 x 4
                nn.Conv2d(nf * 8, 1, 4, 1, 0),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if self.gan_type == 'gan':
            return self.sigmoid(self.main(input).view(-1))
        else:
            return self.main(input).view(-1)

class discriminator_32x32(nn.Module):
    def __init__(self, z_dim, nc=1, nf=64, gan_type='gan'):
        super(discriminator_32x32, self).__init__()
        self.gan_type = gan_type
        self.main = nn.Sequential(
                # input is (nc) x 32 x 32
                dcgan_conv(nc, nf),
                # state size. (nf) x 16 x 16
                dcgan_conv(nf, nf * 2),
                # state size. (nf*2) x 8 x 8
                dcgan_conv(nf * 2, nf * 4),
                # state size. (nf*4) x 4 x 4
                dcgan_conv(nf * 4, nf * 8),
                # state size. (nf*8) x 2 x 2
                nn.Conv2d(nf * 8, 1, 2, 1, 0),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if self.gan_type == 'gan':
            return self.sigmoid(self.main(input).view(-1))
        else:
            return self.main(input).view(-1)

class generator_64x64(nn.Module):
    def __init__(self, z_dim, nc=1, nonlinearity='sigmoid'):
        super(generator_64x64, self).__init__()
        nf = 64
        self.z_dim = z_dim
        if nonlinearity == 'sigmoid':
            self.outfunc = nn.Sigmoid()
        elif nonlinearity == 'tanh': 
            self.outfunc = nn.Tanh()
        elif nonlinearity == 'linear':
            self.outfunc = None
        else:
            raise ValueError('Unknown nonlinearity %s' % nonlinearity)
        self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(z_dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (nf*8) x 4 x 4
                dcgan_upconv(nf * 8, nf * 4),
                # state size. (nf*4) x 8 x 8
                dcgan_upconv(nf * 4, nf * 2),
                # state size. (nf*2) x 16 x 16
                dcgan_upconv(nf * 2, nf),
                # state size. (nf) x 32 x 32
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        if self.outfunc:
            return self.outfunc(self.main(input))
        else:
            return self.main(input)

class generator_32x32(nn.Module):
    def __init__(self, z_dim, nc=1, nonlinearity='sigmoid'):
        super(generator_32x32, self).__init__()
        nf = 64
        self.z_dim = z_dim
        if nonlinearity == 'sigmoid':
            self.outfunc = nn.Sigmoid()
        elif nonlinearity == 'tanh': 
            self.outfunc = nn.Tanh()
        elif nonlinearity == 'linear':
            self.outfunc = None
        else:
            raise ValueError('Unknown nonlinearity %s' % nonlinearity)
        self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(z_dim, nf * 8, 2, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (nf*8) x 2 x 2
                dcgan_upconv(nf * 8, nf * 4),
                # state size. (nf*4) x 4 x 4
                dcgan_upconv(nf * 4, nf * 2),
                # state size. (nf*2) x 8 x 8
                dcgan_upconv(nf * 2, nf),
                # state size. (nf) x 16 x 16
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                # state size. (nc) x 32 x 32
                )

    def forward(self, input):
        if self.outfunc:
            return self.outfunc(self.main(input))
        else:
            return self.main(input)


class vae_32x32(nn.Module):
    def __init__(self, z_dim, y_dim, nc=1, nf=64):
        super(vae_32x32, self).__init__()
        self.encoder = nn.Sequential(
                # input is (nc) x 32 x 32
                dcgan_conv(nc, nf),
                # state size. (nf) x 16 x 16
                dcgan_conv(nf, nf * 2),
                # state size. (nf*2) x 8 x 8
                dcgan_conv(nf * 2, nf * 4),
                # state size. (nf*4) x 4 x 4
                dcgan_conv(nf * 4, nf * 8),
                # state size. (nf*8) x 2 x 2
        )
        self.mu_net = nn.Conv2d(nf * 8, z_dim, 2, 1, 0),
        self.logvar_net = nn.Conv2d(nf * 8, z_dim, 2, 1, 0),

        self.decoder = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(z_dim+y_dim, nf * 8, 2, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (nf*8) x 2 x 2
                dcgan_upconv(nf * 8, nf * 4),
                # state size. (nf*4) x 4 x 4
                dcgan_upconv(nf * 4, nf * 2),
                # state size. (nf*2) x 8 x 8
                dcgan_upconv(nf * 2, nf),
                # state size. (nf) x 16 x 16
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                # state size. (nc) x 32 x 32
                nn.Sigmoid()
                )

    def reparameterize(self, mu, logvar):
        sigma = logvar.mul(0.5).exp_()
        epsilon = logvar.data.new(logvar.size()).normal_()
        z = epsilon.mul(sigma) + mu
        return z

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def decode(self, z, y):
        return self.decoder(torch.cat([z, y], 1))

    def forward(self, input):
        x, y = input # image, labels
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y)
        return x_hat, mu, logvar

        
