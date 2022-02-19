import torch
import generative_model as gm

# Section 3.1/3.2: 2D circles -- 4 ensembles sizes, 10 chains each -> 40 GANs in total
ndim = 2
epochs = [100000,33333,20000,200]
torch.manual_seed(0)
for i,nsamples in enumerate([200,600,1000,100000]):
    for nchain in range(10):
        model = gm.GAN(ndim)
        real_samples = gm.get_circle_data(ndim=ndim,nsamples=nsamples,chain=nchain)
        model.optimize(real_samples,lr=1e-4,epochs=epochs[i],nbatch=50,betas=(0.5,0.999),smooth=True)
        torch.save(model,'trained_gans/circle_samples_'+str(nsamples)+'_chain_'+str(nchain)+'.pt')

# Section 3.3: Hyperspheres -- 2D, 3D, 4D, 7D, 10D -> 5 GANs in total
torch.manual_seed(0)
for ndim in [2,3,4,7,10]:
    model = gm.GAN(ndim)
    real_samples = gm.get_circle_data(ndim=ndim)
    model.optimize(real_samples,lr=1e-4,epochs=500,nbatch=100,betas=(0.9,0.999))
    torch.save(model,'trained_gans/circle_dim_'+str(ndim)+'.pt')
