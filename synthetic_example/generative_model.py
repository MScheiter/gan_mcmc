import torch
from torch import nn

import tqdm
import pickle

import neural_networks as nns

def get_circle_data(ndim=2,nsamples=100000,chain=0):
    if ndim == 2:
        if nsamples == 100000:
            with open(r'datasets/mcmc_samples_pcp_dim_2.obj','rb') as file:
                data = pickle.load(file)
        else:
            with open(r'datasets/mcmc_samples_pcp_dim_2_reduced_'+str(nsamples)+'.obj','rb') as file:
                data = pickle.load(file)
        data = data[:,chain,:]
    else:
        with open(r'datasets/mcmc_samples_pcp_dim_'+str(ndim)+'.obj','rb') as file:
            data = pickle.load(file)
    return torch.Tensor(data)


class GAN(nn.Module):
    def __init__(self,ndim,nhid=50,nlatent=100):
        super().__init__()

        self.ndim = ndim
        self.nlatent = nlatent

        # Build Generator
        layer1 = nns.create_layer_dict(self.nlatent,nhid,normalize=False,dropout=0.4,activation='leakyrelu')
        layer2 = nns.create_layer_dict(nhid,nhid,normalize=False,dropout=0.4,activation='leakyrelu')
        layer3 = nns.create_layer_dict(nhid,nhid,normalize=False,dropout=0.4,activation='leakyrelu')
        layer4 = nns.create_layer_dict(nhid,self.ndim,normalize=False,dropout=0.0,activation=None)
        gen_layers = [layer1,layer2,layer3,layer4]
        self.gen = nns.mlp(gen_layers)

        # Build Discriminator
        layer1 = nns.create_layer_dict(self.ndim,nhid,normalize=False,dropout=0.0,activation='leakyrelu')
        layer2 = nns.create_layer_dict(nhid,nhid,normalize=False,dropout=0.0,activation='leakyrelu')
        layer3 = nns.create_layer_dict(nhid,nhid,normalize=False,dropout=0.0,activation='leakyrelu')
        layer4 = nns.create_layer_dict(nhid,1,normalize=False,dropout=0.0,activation='sigmoid')
        disc_layers = [layer1,layer2,layer3,layer4]
        self.disc = nns.mlp(disc_layers)


    def optimize(self,data,lr=1e-4,betas=(0.9,0.999),epochs=500,nbatch=100,kkd=1,kkg=1,smooth=False):

        nsamples = data.shape[0]

        optimizer_gen = torch.optim.Adam(self.gen.parameters(),lr=lr,betas=betas)
        optimizer_disc = torch.optim.Adam(self.disc.parameters(),lr=lr,betas=betas)

        loss = torch.nn.BCELoss()

        one_labels = 1.0*torch.ones(nbatch)
        zero_labels = 0.0*torch.zeros(nbatch)

        iterations_per_epoch = int(nsamples/nbatch)

        for epoch in tqdm.tqdm(range(epochs)):

            data = data[torch.randperm(nsamples)]

            for i in range(iterations_per_epoch):

                ### Update Discriminator
                for k in range(kkd):
                    optimizer_disc.zero_grad()
                    real_data = data[i*nbatch:(i+1)*nbatch]
                    if smooth:
                        real_data = real_data + 0.1 * torch.randn(nbatch,self.ndim)
                    fake_data = self.gen(torch.randn(nbatch,self.nlatent))
                    score_real = loss(self.disc(real_data).squeeze(),one_labels)
                    score_fake = loss(self.disc(fake_data).squeeze(),zero_labels)
                    score_disc = score_real + score_fake
                    score_disc.backward()
                    optimizer_disc.step()

                ### Update Generator
                for k in range(kkg):
                    optimizer_gen.zero_grad()
                    fake_data = self.gen(torch.randn(nbatch,self.nlatent))
                    score_gen = loss(self.disc(fake_data).squeeze(),one_labels)
                    score_gen.backward()
                    optimizer_gen.step()


class WGAN(nn.Module):
    def __init__(self,ndim,nhid=200,nlatent=100):
        super().__init__()

        self.ndim = ndim
        self.nlatent = nlatent
        self.clip_value = 0.01

        # Build Generator
        layer1 = nns.create_layer_dict(self.nlatent,nhid,normalize=False,dropout=0.4,activation='leakyrelu')
        layer2 = nns.create_layer_dict(nhid,nhid,normalize=False,dropout=0.4,activation='leakyrelu')
        layer3 = nns.create_layer_dict(nhid,nhid,normalize=False,dropout=0.4,activation='leakyrelu')
        layer4 = nns.create_layer_dict(nhid,ndim,normalize=False,dropout=0.0,activation=None)
        gen_layers = [layer1,layer2,layer3,layer4]
        self.gen = nns.mlp(gen_layers)

        # Build Discriminator
        layer1 = nns.create_layer_dict(ndim,nhid,normalize=False,dropout=0.0,activation='leakyrelu')
        layer2 = nns.create_layer_dict(nhid,nhid,normalize=False,dropout=0.0,activation='leakyrelu')
        layer3 = nns.create_layer_dict(nhid,nhid,normalize=False,dropout=0.0,activation='leakyrelu')
        layer4 = nns.create_layer_dict(nhid,1,normalize=False,dropout=0.0,activation=None)
        disc_layers = [layer1,layer2,layer3,layer4]
        self.disc = nns.mlp(disc_layers)


    def optimize(self,data,lr=1e-4,epochs=200,nbatch=100,kkd=1,kkg=1):

        nsamples = data.shape[0]

        optimizer_gen = torch.optim.RMSprop(self.gen.parameters(),lr=lr)
        optimizer_disc = torch.optim.RMSprop(self.disc.parameters(),lr=lr)

        iterations_per_epoch = int(nsamples/nbatch)

        for epoch in tqdm.tqdm(range(epochs)):

            data = data[torch.randperm(nsamples)]

            for i in range(iterations_per_epoch):

                ### Update Discriminator
                for k in range(kkd):
                    optimizer_disc.zero_grad()

                    real_data = data[i*nbatch:(i+1)*nbatch]
                    fake_data = self.gen(torch.randn(nbatch,self.nlatent))

                    score_disc = -torch.mean(self.disc(real_data)) + torch.mean(self.disc(fake_data))

                    score_disc.backward()
                    optimizer_disc.step()

                    for p in self.disc.parameters():
                        p.data.clamp_(-self.clip_value,self.clip_value)

                ### Update Generator
                for k in range(kkg):
                    optimizer_gen.zero_grad()

                    fake_data = self.gen(torch.randn(nbatch,self.nlatent))
                    score_gen = -torch.mean(self.disc(fake_data))
                    score_gen.backward()

                    optimizer_gen.step()





class DCGAN(nn.Module):
    def __init__(self,nlatent=100,ngf=64,ndf=64):
        super().__init__()

        self.nlatent = nlatent

        self.gen = nn.Sequential(
            # Input: latent vector of length nlatent
            nn.ConvTranspose2d(self.nlatent, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Size: (ngf*8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Size: (ngf*4) x 4 x 4
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Size: (ngf*2) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Size: (ngf) x 16 x 16
            nn.ConvTranspose2d( ngf, 1, 3, 2, 5, bias=False),
            nn.Tanh()
            # Output: 1 x 23 x 23
        )

        self.disc = nn.Sequential(
            # Input: 1 x 23 x 23
            nn.Conv2d(1, ndf, 6, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ndf*8) x 2 x 2
            nn.Conv2d(ndf * 8, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

        self.gen.apply(self.weights_init)
        self.disc.apply(self.weights_init)


    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    def optimize(self,data,lr=1e-4,epochs=10,nbatch=128,betas=(0.5,0.999),kkd=1,kkg=1):

        nsamples = data.shape[0]

        optimizer_gen = torch.optim.Adam(self.gen.parameters(),lr=lr,betas=betas)
        optimizer_disc = torch.optim.Adam(self.disc.parameters(),lr=lr,betas=betas)

        loss = torch.nn.BCELoss()

        one_labels = 1.0*torch.ones(nbatch)
        zero_labels = 0.0*torch.zeros(nbatch)

        iterations_per_epoch = int(nsamples/nbatch)

        for epoch in range(epochs):
            print('epoch '+str(epoch)+'/'+str(epochs))

            data = data[torch.randperm(nsamples)]

            for i in tqdm.tqdm(range(iterations_per_epoch)):

                ### Update Discriminator
                for k in range(kkd):
                    optimizer_disc.zero_grad()

                    real_data = data[i*nbatch:(i+1)*nbatch,:].unsqueeze(1)
                    score_real = loss(self.disc(real_data).squeeze(),one_labels)
                    score_real.backward()

                    fake_data = self.gen(torch.randn(nbatch,self.nlatent,1,1))
                    score_fake = loss(self.disc(fake_data).squeeze(),zero_labels)
                    score_fake.backward()

                    optimizer_disc.step()

                ### Update Generator
                for k in range(kkg):
                    optimizer_gen.zero_grad()

                    fake_data = self.gen(torch.randn(nbatch,self.nlatent,1,1))
                    score_gen = loss(self.disc(fake_data).squeeze(),one_labels)
                    score_gen.backward()

                    optimizer_gen.step()
