import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

dpi = 280

##### Figure 1 -- Motivating example with simple Gaussian distribution #####

np.random.seed(42)

def gauss(x,mu=0,sigma=1):
    return 1/np.sqrt(2*np.pi)/sigma * np.exp(-0.5*(x-mu)**2/sigma**2)

def add_label(ax,plot_number):
    label = '('+chr(96+plot_number)+')'
    ax.text(-0.02,1,label,color='k',transform=ax.transAxes,ha='right',va='top',fontweight='bold',fontsize=13)

bins = np.linspace(-4,4,29)
bins_cont = np.linspace(-4,4,100)

col1 = 'rebeccapurple'

# Generate a collection of 'observed' samples
n_plot = 30
samples = np.random.normal(0,1,n_plot)

fig = plt.figure(figsize=(13,4))
for i in range(2):
    ax = fig.add_subplot(1,3,i+1)
    add_label(ax,i+1)

    # Fig. a + b: Plot observed samples
    hist = ax.hist(samples,bins=bins,color=[0.9,0.9,0.9],ec='k',linewidth=0.5)

    if i==1:
        # Only Fig. b: Calculate and plot Gaussian approximation
        samples_mean,samples_std = np.mean(samples),np.std(samples)
        gauss_approx = gauss(bins_cont,samples_mean,samples_std)
        gauss_approx = gauss_approx / np.max(gauss_approx) * np.max(hist[0])
        ax.plot(bins_cont,gauss_approx,color=col1,linewidth=3)
        ax.text(0.05,0.96,'Mean: '+str(round(samples_mean,1))+', Std: '+str(round(samples_std,1)),color=col1,fontsize=13,transform=ax.transAxes,ha='left',va='top',bbox=dict(facecolor=[0.9,0.9,0.9],alpha=0.5,ec=col1,linewidth=0.5))
    else:
        ax.text(0.05,0.96,str(n_plot)+' Samples',color='k',fontsize=13,transform=ax.transAxes,ha='left',va='top',bbox=dict(facecolor=[0.9,0.9,0.9],alpha=0.9,linewidth=0.5))

    ax.set_xlim(-4,4)
    ax.set_ylim(0,7)
    ax.set_yticks([])

# Draw many samples from Gaussian approximation
samples_new = np.random.normal(samples_mean,samples_std,10000)

ax = fig.add_subplot(1,3,3)
add_label(ax,3)

# Fig. c: Gaussian approximation with histogram of many samples drawn from it
hist = ax.hist(samples_new,bins=bins,density=True,color=[0.9,0.9,0.9],ec=col1,linewidth=0.5)
ax.text(0.05,0.96,str(10000)+' Samples',color=col1,fontsize=13,transform=ax.transAxes,ha='left',va='top',bbox=dict(facecolor=[0.9,0.9,0.9],alpha=0.5,ec=col1,linewidth=0.8))
ax.plot(bins_cont,gauss_approx/gauss_approx.max()*hist[0].max(),color=col1,linewidth=3)
ax.set_xlim(-4,4)
ax.set_ylim(0,.53)
ax.set_yticks([])

plt.tight_layout()

fig.savefig('../figures/fig01.png',dpi=dpi)


##### Figure 2 -- Comparison of GAN and McMC circles for different ensemble sizes #####

# Plot settings
bins = np.linspace(-1,1,51)
colors = ['k','rebeccapurple']
cmaps = ['binary','Purples']
labels = ['McMC Samples','GAN Samples']

# Create figure
width = 7*4 - 1
height = 7*2 - 1
fig = plt.figure(figsize=(0.4*width,0.4*height))

# Iterate through all ensemble sizes for McMC and GAN
for i,n_samples in enumerate([200,600,1000,100000]):
    for j,what in enumerate(['real','fake']):

        # Load samples
        with open(r'processed_data/'+what+'_'+str(n_samples)+'.obj','rb') as file:
            data = pickle.load(file)
        data = data[:,7,:]

        # Prepare subplots for joint and marginal distributions
        start_x = i*7
        start_y = j*7
        ax0 = plt.subplot2grid((height,width),(start_y,start_x),colspan=5)
        ax1 = plt.subplot2grid((height,width),(start_y+1,start_x),colspan=5,rowspan=5)
        ax2 = plt.subplot2grid((height,width),(start_y+1,start_x+5),rowspan=5)

        # Plot first marginal (horizontal)
        ax0.hist(data[:,0],bins=bins,color=colors[j],ec='gray')
        ax0.set_xlim(-1,1)
        ax0.axis('off')

        # Plot joint distribution
        ax1.hist2d(data[:,0],data[:,1],bins=bins,cmap=cmaps[j])
        ax1.set_xlim(-1,1)
        ax1.set_ylim(-1,1)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Plot second marginal (vertical)
        ax2.hist(data[:,1],bins=bins,color=colors[j],ec='gray',orientation='horizontal')
        ax2.set_ylim(-1,1)
        ax2.axis('off')

        # Add figure labels where required
        if i == 0:
            ax1.set_ylabel(labels[j])
        if j == 0:
            if i < 3:
                ax0.set_title(str(n_samples)+' Samples')
            else:
                ax0.set_title('100k Samples')

fig.savefig('../figures/fig02.png',dpi=dpi,bbox_inches='tight')


##### Figure 3 -- GAN-enhanced numerical integration #####

def text_in_corner(ax,text):
    ax.text(0.05,0.96,text,color='k',fontsize=10,transform=ax.transAxes,ha='left',va='top',bbox=dict(facecolor=[0.9,0.9,0.9],alpha=0.9,linewidth=0.5))

# Define function for integral
def f1(phi,n):
    return np.sin(n*phi)

# Conversion from Cartesian to angle
def xy2phi(x,y):
    return np.arctan2(x,y) % (2*np.pi)

# Define grid to plot function
x = np.linspace(-1,1,1000)
y = np.linspace(-1,1,1000)
X,Y = np.meshgrid(x,y)
phi = xy2phi(X,Y)

# Set up figure
axs = []
fig = plt.figure(figsize=(9,6))
gs = gridspec.GridSpec(1,3,wspace=0.1)
gs.update(left=0.07,right=0.9,top=0.98,bottom=0.52)
axs.append(plt.subplot(gs[0]))
axs.append(plt.subplot(gs[1]))
axs.append(plt.subplot(gs[2]))
gs = gridspec.GridSpec(1,1)
gs.update(left=0.92,right=0.98,top=0.95,bottom=0.55)
axs.append(plt.subplot(gs[0]))
gs = gridspec.GridSpec(1,2,wspace=0.125)
gs.update(left=0.07,right=0.98,top=0.48,bottom=0.1)
axs.append(plt.subplot(gs[0]))
axs.append(plt.subplot(gs[1]))

# Plot three examples of function
for i,n in enumerate([3.5,10.5,24.5]):
    f = f1(phi,n)
    im = axs[i].imshow(f,cmap='RdBu')
    axs[i].axis('off')
    text_in_corner(axs[i],'freq = '+str(n))
cbar = plt.colorbar(im,ax=axs[3],orientation='vertical',fraction=0.9,shrink=0.4,aspect=6)
axs[3].axis('off')
cbar.set_ticks([-1,0,1])

# Plot integral stds of McMC and GAN
with open(r'processed_data/integrals_std_array.obj','rb') as file:
    std_array = pickle.load(file)

colors = ['k','rebeccapurple']
alphas = 0.2,0.6,0.8,1
freqs = np.arange(100)+0.5
titles = ['MC Integration','GAN-enhanced MC Integration']

for i,what in enumerate(['real','fake']):
    for j,n_samples in enumerate([200,600,1000,100000]):
        axs[i+4].plot(freqs,std_array[i,j,:].T,color=colors[i],alpha=alphas[j],label=str(n_samples))
    axs[i+4].set_xlim(0,100)
    axs[i+4].set_ylim(0,21)
    axs[i+4].legend(loc='upper left',title='Ensemble size',fancybox=True,fontsize=10,title_fontsize=11)
    if i==0:
        axs[i+4].set_ylabel('Std of relative error',fontsize=12)
    axs[i+4].set_xlabel('Frequency',fontsize=12)
    axs[i+4].set_title(titles[i],fontsize=12)

fig.savefig('../figures/fig03.png',dpi=dpi)


##### Figure 4 -- Curse of dimensionality #####

def text_in_corner(ax,text,color,edgecolor):
    ax.text(0.05,0.95,text,color=color,transform=ax.transAxes,ha='left',va='top',bbox=dict(facecolor=[0.9,0.9,0.9],alpha=0.7,linewidth=0.8,edgecolor=edgecolor))

def add_ylabel(ax,label):
    ax.set_ylabel(label,fontsize=12)

# Function to extract data samples near the plane
def get_slice(data):
    dim = data.shape[1]
    for i in range(dim-2):
        data = data[np.abs(data[:,dim-1])<0.15,:dim-1]
        dim -= 1
    return data

# General function for one individual subplot
def make_one_plot(ax,dim,n_plot):

    # For first two plots use McMC ensemble
    if n_plot < 2:
        with open('datasets/mcmc_samples_pcp_dim_'+str(dim)+'.obj','rb') as file:
            data = pickle.load(file)
            # in 2D combine first 3 chains
            if dim==2:
                data = data[:,:3,:].reshape(-1,dim)

        # For second plot get samples near plane
        if n_plot == 1:
            data = get_slice(data)

    # For third plot use GAN ensemble
    if n_plot == 2:
        with open('processed_data/sliced_samples_dim_'+str(dim)+'.obj','rb') as file:
            data = pickle.load(file)

    # Plot ensemble
    ax.hist2d(data[:,0],data[:,1],cmap=cmaps[n_plot],bins=np.linspace(-1,1,100))

    text_in_corner(ax,'N='+str(data.shape[0]),color=colors[n_plot],edgecolor=colors[n_plot])
    ax.set_yticks([])
    ax.set_xticks([])

labels = ['McMC (all samples)',f'McMC ($x_1-x_2$ plane)',f'GAN ($x_1-x_2$ plane)']
cmaps = ['binary','binary','Purples']
colors = ['k','k','rebeccapurple']

dims = 2,3,4,7,10

fig = plt.figure(figsize=(2*len(dims),6.1))

# Iterate through dimensions
for i,dim in enumerate(dims):
    axs = []
    # Iterate through the three types of plots
    for j in range(3):
        axs.append(fig.add_subplot(3,len(dims),i+j*len(dims)+1))
        make_one_plot(axs[j],dim,n_plot=j)

        if i==0:
            add_ylabel(axs[j],labels[j])

    axs[0].set_title(str(dim)+'D',fontsize=12)

plt.tight_layout()

fig.savefig('../figures/fig04.png',dpi=dpi)
