import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec,colors
import cartopy.crs as ccrs
import pickle

dpi = 500

### Load data

data_path = 'plotting_data/'

def get_data_dict(perc=100):
    if perc == 100:
        data_dict_object = 'data_dict_100perc.obj'
    if perc == 40:
        data_dict_object = 'data_dict_40perc.obj'
    if perc == 50:
        data_dict_object = 'data_dict_50perc.obj'
    if perc == -99:
        data_dict_object = 'data_dict_patches.obj'

    data_dict_file = data_path + data_dict_object
    with open(data_dict_file,'rb') as file:
        data_dict = pickle.load(file)

    if perc == 100:
        with open(data_path+'real_marginals.obj','rb') as file:
            data_dict['chosen_real'] = pickle.load(file)
        with open(data_path+'fake_marginals.obj','rb') as file:
            data_dict['chosen_fake'] = pickle.load(file)

    return data_dict

data_dict = get_data_dict()

### Plot settings

lon_center = 131
mean_min,mean_max = 6.8,7.6
std_min,std_max = 0,0.3
skew_min,skew_max = -2.5,2.5
kurt_min,kurt_max = -2,10
cov_min,cov_max = -0.01,0.01
labelpad = -44

cmap = plt.get_cmap('RdBu')
cmap_std = plt.get_cmap('Blues')

divnorm = colors.TwoSlopeNorm(vmin=kurt_min,vcenter=0,vmax=kurt_max)

### Plot functions
def plot_contour(data,colormap,min,max,pos=None,title=None,ylabel=None,twoslopes=False,rowspan=3,colspan=2,extend='neither',noaxis=True,nlevels=50):
    if type(pos)==tuple:
        ax = plt.subplot2grid(subplot_size,pos,rowspan=rowspan,colspan=colspan,projection=ccrs.Mollweide(central_longitude=lon_center))
    else:
        ax = plt.subplot(pos[1:,:],projection=ccrs.Mollweide(central_longitude=lon_center))
    if twoslopes:
        pcm = ax.contourf(data_dict['lon'],data_dict['lat'],data,transform=ccrs.PlateCarree(),cmap=colormap,levels=np.linspace(min,max,nlevels),extend=extend,norm=divnorm)
    else:
        pcm = ax.contourf(data_dict['lon'],data_dict['lat'],data,transform=ccrs.PlateCarree(),cmap=colormap,levels=np.linspace(min,max,nlevels),extend=extend)
    ax.coastlines()
    if noaxis:
        ax.axis('off')
    if title:
        ax.set_title(title,fontsize=12)
    if ylabel:
        add_ylabel(ax,ylabel,fontsize=15)
    return ax,pcm


def add_ylabel(ax,label,**kwargs):
    ax.text(-0.2,0.5,label,va='center',ha='center',
            rotation='horizontal',rotation_mode='anchor',
            transform=ax.transAxes,**kwargs)

def add_colorbar(pcm,min,max,pos=None,label=None,vertical=False,fraction=0.3,shrink=0.6,aspect=12,rowspan=1,round=2):
    if vertical:
        cax = plt.subplot2grid(subplot_size,pos,rowspan=3)
        cbar = plt.colorbar(pcm,ax=cax,orientation='vertical',fraction=fraction,shrink=shrink,aspect=10)
        cbar.set_label(label,labelpad=-60,fontsize=12)
    else:
        if type(pos)==tuple:
            cax = plt.subplot2grid(subplot_size,pos,colspan=2,rowspan=rowspan)
        else:
            cax = plt.subplot(pos[0,:])
        cbar = plt.colorbar(pcm,ax=cax,orientation='horizontal',fraction=fraction,shrink=shrink,aspect=aspect)
        cbar.set_label(label,labelpad=labelpad,fontsize=15)
        cbar.ax.tick_params(labelsize=14)

    cax.axis('off')
    cbar.set_ticks([np.round(min,round),np.round(max,round),np.round((max+min)/2,round)])
    cbar.set_ticklabels([np.round(min,round),np.round(max,round),np.round((max+min)/2,round)])
    return cax

def plot_marginal(ax,i,j):
    ax.hist(data_dict['chosen_real'][:,j*4+i],bins=bins_x,color='k',ec='gray',alpha=0.5)
    ax.hist(data_dict['chosen_fake'][:,j*4+i],bins=bins_x,color='rebeccapurple',ec='gray',alpha=0.7)
    if i==0 and j==4:
        ax.set_xticks([6.8,7.2,7.6])
        ax.set_xlabel('Vs [km/s]')
        ax.text(0.98,0.95,'McMC',va='top',ha='right',
                rotation='horizontal',rotation_mode='anchor',
                transform=ax.transAxes)
        ax.text(0.98,0.8,'GAN',va='top',ha='right',color='rebeccapurple',
                rotation='horizontal',rotation_mode='anchor',
                transform=ax.transAxes)
    else:
        ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(bins_x.min(),bins_x.max())
    emd = data_dict['emds'].flatten()[data_dict['chosen_ind_flat']][j*4+i]
    emd = np.round(emd,3)
    ax.text(0.02,0.95,str(emd),va='top',ha='left',color='dodgerblue',
            rotation='horizontal',rotation_mode='anchor',
            transform=ax.transAxes)

    if i==0 and j==0:
        add_label(ax,3)


##### Figure 5 -- First four moments for McMC and GAN #####

fig = plt.figure(figsize=(10.5,6.5))
subplot_size = (7,8)

# Plot McMC data and colorbars
ax,pcm = plot_contour(data_dict['real_mean'],cmap,mean_min,mean_max,pos=(1,0),ylabel='McMC')
add_colorbar(pcm,mean_min,mean_max,pos=(0,0),label='Mean')
ax,pcm = plot_contour(data_dict['real_std'],cmap_std,std_min,std_max,pos=(1,2))
add_colorbar(pcm,std_min,std_max,pos=(0,2),label='Std. Dev.')
ax,pcm = plot_contour(data_dict['real_skew'],cmap,skew_min,skew_max,pos=(1,4))
add_colorbar(pcm,skew_min,skew_max,pos=(0,4),label='Skewness')
ax,pcm = plot_contour(data_dict['real_kurt'],cmap,kurt_min,kurt_max,pos=(1,6),twoslopes=True)
add_colorbar(pcm,kurt_min,kurt_max,pos=(0,6),label='Kurtosis')

# Plot GAN data
ax,pcm = plot_contour(data_dict['fake_mean'],cmap,mean_min,mean_max,pos=(4,0),ylabel='GAN')
ax,pcm = plot_contour(data_dict['fake_std'],cmap_std,std_min,std_max,pos=(4,2))
ax,pcm = plot_contour(data_dict['fake_skew'],cmap,skew_min,skew_max,pos=(4,4))
ax,pcm = plot_contour(data_dict['fake_kurt'],cmap,kurt_min,kurt_max,pos=(4,6),twoslopes=True)

plt.tight_layout()

fig.savefig('../figures/fig05.png',dpi=dpi)


##### Figure 6 -- Marginals and EMDs #####

bins_x = np.linspace(mean_min-0.2,mean_max+0.2,50)
bins_emd = np.linspace(0,0.1,50)

def add_label(ax,plot_number,leftshift=0,upshift=0):
    label = '('+chr(96+plot_number)+')'
    ax.text(-0.05-leftshift,1+upshift,label,color='k',transform=ax.transAxes,ha='right',va='top',fontweight='bold',fontsize=12)

fig = plt.figure(figsize=(11.5,6.5))

# Plot map of earth mover distance (EMD) with specific locations and color map
shift = 0.02
gs = gridspec.GridSpec(6,1)
gs.update(left=0.02+shift,right=1/3-shift,top=0.98,bottom=0.22)
ax,pcm = plot_contour(data_dict['emds'],cmap_std,0,0.1,pos=gs)
ax.plot(data_dict['chosen_lon'],data_dict['chosen_lat'],'x',color='rebeccapurple',markersize=6,transform=ccrs.PlateCarree())
ax = add_colorbar(pcm,0,0.1,pos=gs,label='Earth Mover Distance',rowspan=2,aspect=12,shrink=0.8,fraction=0.2)
add_label(ax,1)

# Plot histogram of EMDs
shift = 0.04
gs = gridspec.GridSpec(1,1)
gs.update(left=0.02+shift,right=1/3-shift,top=0.2,bottom=0.07)
ax = plt.subplot(gs[:,:])
ax.hist(data_dict['emds'].flatten(),bins=bins_emd,color='dodgerblue',ec='k',density=True)
ax.set_xlabel('EMD')
ax.set_ylabel('Frequency')
ax.set_xlim(0,0.1)
add_label(ax,2,leftshift=0.1,upshift=0.15)

# Plot marginals at specific locations
gs = gridspec.GridSpec(5,4,wspace=0.1,hspace=0.1)
gs.update(left=0.35,right=0.98,top=0.98,bottom=0.07)
for j in range(5):
    for i in range(4):
        ax = plt.subplot(gs[j,i])
        plot_marginal(ax,i,j)

fig.savefig('../figures/fig06.png',dpi=dpi)


##### Figure 7 -- Covariance Matrices and point spread functions #####

# Set up figure
axs = []
fig = plt.figure(figsize=(16,6.5))

gs = gridspec.GridSpec(2,1,wspace=0.1,hspace=0.1)
gs.update(left=0.06,right=0.23,top=0.94,bottom=0.04)
axs.append(plt.subplot(gs[0],projection=ccrs.Mollweide(central_longitude=lon_center)))
axs.append(plt.subplot(gs[1],projection=ccrs.Mollweide(central_longitude=lon_center)))

gs = gridspec.GridSpec(2,4,wspace=0.1,hspace=0.1)
gs.update(left=0.26,right=0.90,top=0.94,bottom=0.04)
for i in range(8):
    axs.append(plt.subplot(gs[i],projection=ccrs.Mollweide(central_longitude=lon_center)))

gs = gridspec.GridSpec(1,1)
gs.update(left=0.93,right=0.98,top=0.8,bottom=0.2)
axs.append(plt.subplot(gs[0]))

im = axs[0].imshow(data_dict['real_cov'],cmap=cmap,vmin=cov_min,vmax=cov_max)
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].text(-0.17,0.5,'McMC',va='center',ha='center',rotation='horizontal',rotation_mode='anchor',transform=axs[0].transAxes,fontsize=15)
axs[0].set_title('Cov. Matrix',fontsize=15)

im = axs[1].imshow(data_dict['fake_cov'],cmap=cmap,vmin=cov_min,vmax=cov_max)
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].text(-0.17,0.5,'GAN',va='center',ha='center',rotation='horizontal',rotation_mode='anchor',transform=axs[1].transAxes,fontsize=15)

indices = 5,6,13,14
for i,index in enumerate(indices):

    real_cov_point = data_dict['real_cov'][:,data_dict['chosen_ind_flat'][index]].reshape(23,23)
    fake_cov_point = data_dict['fake_cov'][:,data_dict['chosen_ind_flat'][index]].reshape(23,23)

    pcm = axs[i+2].contourf(data_dict['lon'],data_dict['lat'],real_cov_point,transform=ccrs.PlateCarree(),cmap=cmap,levels=np.linspace(cov_min,cov_max,50),extend='both')
    axs[i+2].coastlines()
    axs[i+2].axis('off')
    axs[i+2].plot(data_dict['chosen_lon'][index],data_dict['chosen_lat'][index],'x',color='white',markersize=8,transform=ccrs.PlateCarree())

    pcm = axs[i+6].contourf(data_dict['lon'],data_dict['lat'],fake_cov_point,transform=ccrs.PlateCarree(),cmap=cmap,levels=np.linspace(cov_min,cov_max,50),extend='both')
    axs[i+6].coastlines()
    axs[i+6].axis('off')
    axs[i+6].plot(data_dict['chosen_lon'][index],data_dict['chosen_lat'][index],'x',color='white',markersize=8,transform=ccrs.PlateCarree())

axs[4].text(0,1.05,'Point spread functions',va='center',ha='center',rotation='horizontal',rotation_mode='anchor',transform=axs[4].transAxes,fontsize=15)

cbar = plt.colorbar(pcm,ax=axs[10],orientation='vertical',fraction=0.9,shrink=0.4,aspect=6)
axs[10].axis('off')
cbar.set_ticks([cov_min,0,cov_max])
cbar.set_label('Covariance',labelpad=-75,fontsize=15)

fig.savefig('../figures/fig07.png',dpi=dpi)



##### Figure 8 -- Convergence assessment with covariance matrices #####

fig = plt.figure(figsize=(8.0,10))

# Loop to repeat the same for both percentages
for i,perc in enumerate([40,50]):
    if i == 0:
        left,right = 0.0,0.51
        extend = 'max'
    else:
        left,right = 0.49,1.0
        extend = 'neither'

    data_dict = get_data_dict(perc)

    # Plot map of EMD and colorbar
    shift = 0.06
    gs = gridspec.GridSpec(6,1)
    gs.update(left=left+shift,right=right-shift,top=1.0,bottom=0.5)
    ax,pcm = plot_contour(data_dict['emds'],cmap_std,0,0.1,pos=gs,extend=extend,nlevels=20)
    title = 'EMD (first '+str(perc)+'%)'
    add_colorbar(pcm,0,0.1,pos=gs,label=title,rowspan=2,aspect=12,shrink=0.8,fraction=0.2)

    # Plot histogram of EMD
    shift = 0.09
    gs = gridspec.GridSpec(1,1)
    gs.update(left=left+shift,right=right-shift,top=0.5,bottom=0.42)
    ax = plt.subplot(gs[:,:])
    ax.hist(data_dict['emds'].flatten(),bins=bins_emd,color='dodgerblue',ec='k',density=True)
    ax.set_xlabel('Earth Mover Distance',fontsize=12)
    ax.set_ylabel('Frequency',fontsize=12)
    ax.set_xlim(0,0.1)
    ax.set_ylim(0,70)

gs = gridspec.GridSpec(1,3,wspace=0.05)
gs.update(left=0.05,right=0.95,top=0.33,bottom=0.05)

for i,perc in enumerate([40,50]):

    data_dict = get_data_dict(perc)
    if i==0:
        ax = plt.subplot(gs[2])
        im = plt.imshow(data_dict['real_cov'],cmap=cmap,vmin=cov_min,vmax=cov_max)
        plt.title('100% (McMC)',fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        add_label(ax,3,leftshift=50)
    ax = plt.subplot(gs[i])
    im = plt.imshow(data_dict['fake_cov'],cmap=cmap,vmin=cov_min,vmax=cov_max)
    plt.title(str(perc)+'% (GAN)',fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])

gs = gridspec.GridSpec(1,1)
gs.update(left=0.25,right=0.75,top=0.045,bottom=0.02)
ax = plt.subplot(gs[:])
cbar = plt.colorbar(im,ax=ax,orientation='horizontal',fraction=0.9,shrink=0.4,aspect=10)
ax.axis('off')
cbar.set_ticks([cov_min,0,cov_max])
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Covariance',labelpad=-45,fontsize=15)

fig.savefig('../figures/fig08.png',dpi=dpi)


##### Figure 9 -- Global map #####

data_dict = get_data_dict(-99)
mean_min,mean_max = 6.8-0.02,7.6
std_min,std_max = 0.0,0.35
nlevels = 18

labelpad = -50

fig = plt.figure(figsize=(12,7))
subplot_size = (7,4)

# Plot maps of McMC data and colorbars
ax,pcm = plot_contour(data_dict['real_mean'],cmap,mean_min,mean_max,pos=(1,0),noaxis=False,nlevels=nlevels)
add_colorbar(pcm,mean_min,mean_max,pos=(0,0),label='Mean',round=1)
ax.text(-0.1,0.5,'McMC',va='center',ha='center',rotation='horizontal',rotation_mode='anchor',transform=ax.transAxes,fontsize=15)
ax,pcm = plot_contour(data_dict['real_std'],cmap_std,std_min,std_max,pos=(1,2),noaxis=False,nlevels=nlevels)
add_colorbar(pcm,std_min,std_max,pos=(0,2),label='Std. Dev.')

# Plot maps of GAN data
ax,pcm = plot_contour(data_dict['fake_mean'],cmap,mean_min,mean_max,pos=(4,0),noaxis=False,nlevels=nlevels)
ax.text(-0.1,0.5,'GAN',va='center',ha='center',rotation='horizontal',rotation_mode='anchor',transform=ax.transAxes,fontsize=15)
ax,pcm = plot_contour(data_dict['fake_std'],cmap_std,std_min,std_max,pos=(4,2),noaxis=False,nlevels=nlevels)

plt.tight_layout()

fig.savefig('../figures/fig09.png',dpi=dpi)
