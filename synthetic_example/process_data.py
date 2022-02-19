import numpy as np
import pickle
import torch

### Extract real and fake data ###

for n_samples in [200,600,1000,100000]:
    if n_samples == 100000:
        with open(r'datasets/mcmc_samples_pcp_dim_2.obj','rb') as file:
            real_data = pickle.load(file)
    else:
        with open(r'datasets/mcmc_samples_pcp_dim_2_reduced_'+str(n_samples)+'.obj','rb') as file:
            real_data = pickle.load(file)

    max_chain = 10
    fake_data = np.zeros((100000,max_chain,2))

    for i in range(max_chain):
        name = 'circle_samples_'+str(n_samples)+'_chain_'+str(i)

        model = torch.load('trained_gans/'+name+'.pt')
        fake_data_chain = model.gen(torch.randn(100000,100)).detach().numpy().astype(np.float64)
        fake_data[:,i,:] = fake_data_chain

    with open(r'processed_data/real_'+str(n_samples)+'.obj','wb') as file:
        pickle.dump(real_data,file)
    with open(r'processed_data/fake_'+str(n_samples)+'.obj','wb') as file:
        pickle.dump(fake_data,file)


### Calculate integrals ###

def xy2phi(x,y):
    return np.arctan2(x,y) % (2*np.pi)
def f1(phi,n):
    return np.sin(n*phi)
def f1_int(n):
    return 2*(np.sin(np.pi*n))**2/n

std_array = np.zeros((2,4,100))

for i,what in enumerate(['real','fake']):
    for j,n_samples in enumerate([200,600,1000,100000]):
        for k,freq in enumerate(np.arange(100)+0.5):

            with open(r'processed_data/'+what+'_'+str(n_samples)+'.obj','rb') as file:
                data = pickle.load(file)

            analyt = f1_int(freq)
            rel_errs = np.zeros(10)

            for l in range(10):
                xy = data[:,l,:]
                phi = xy2phi(xy[:,0],xy[:,1])

                integral = np.mean(f1(phi,freq)) * 2*np.pi
                rel_errs[l] = (integral - analyt) / analyt

            std_array[i,j,k] = np.std(rel_errs)

with open(r'processed_data/integrals_std_array.obj','wb') as file:
    pickle.dump(std_array,file)


### Sample data from high-dimensional GANs ###

def get_slice(data):
    dim = data.shape[1]
    for i in range(dim-2):
        data = data[np.abs(data[:,dim-1])<0.15,:dim-1]
        dim -= 1
    return data

target = int(3e5)
step = int(2e5)

dims = 2,3,4,7,10

for i,dim in enumerate(dims):
    print('-----')
    print(dim)
    slice_data = np.zeros((target,2))
    done = 0

    for j in range(20000):

        if done >= target:
            break

        model = torch.load('trained_gans/circle_dim_'+str(dim)+'.pt')
        data = model.gen(torch.randn(step,100)).detach().numpy()

        data = get_slice(data)

        new = data.shape[0]
        if done+new > target:
            slice_data[done:,:] = data[:(target-done),:]
        else:
            slice_data[done:done+new,:] = data
        done += new

    with open('processed_data/sliced_samples_dim_'+str(dim)+'.obj','wb') as file:
        pickle.dump(slice_data, file)
