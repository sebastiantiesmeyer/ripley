import torch

if torch.cuda.is_available():
    import torch.cuda as t
    device = 'cuda:0'
else:
    import torch as t
    device='cpu'

import numpy as np
import scipy.signal
from tqdm import tqdm_notebook
from skimage import transform  
from tempfile import TemporaryFile
from scipy.special import gammaincc 

def _torch_complex_product(x, y):
    '''Point-wise complex product of tensors x and y (not yet available in native pytorch).'''
    output = torch.zeros(x.shape, device=device)
    output[:,:,0] = x[:,:,0]*y[:,:,0]-x[:,:,1]*y[:,:,1]
    output[:,:,1] = (x[:,:,0]+x[:,:,1])*(y[:,:,0]+y[:,:,1])-x[:,:,0]*y[:,:,0]-x[:,:,1]*y[:,:,1]
    return output




def _get_kernels(max_radius,linear_steps):

    if linear_steps>max_radius: linear_steps=max_radius

    vals = np.ones((max_radius)+1)
    vals[0]=0
    vals[linear_steps+1:]+=np.arange(max_radius-linear_steps)
    vals = np.cumsum(vals)
    vals = vals[:(vals<max_radius).sum()+1]
    vals[-1]=max_radius

    span = np.arange(-max_radius , max_radius +1)
    X,Y = np.meshgrid(span,span)
    dists = (X**2+Y**2)**0.5

    kernel_1 = np.zeros_like(X)
    kernel_1[max_radius,max_radius]=1
    kernels = [kernel_1]

    for i in range(len(vals)-1):

        r1 = vals[i]
        r2 = vals[i+1]
        
        kernel_1 = (dists-r1)
        kernel_1 = -(kernel_1-1)*(kernel_1<1)
        kernel_1[kernel_1>1] = 1
        kernel_1 = 1-kernel_1 

        kernel_2 = (dists-r2)
        kernel_2 = -(kernel_2-1)*(kernel_2<1)
        kernel_2[kernel_2>1] = 1
        kernel_2[kernel_2==1] = kernel_1[kernel_2==1]
        
        kernels.append(kernel_2)
        
    kernels = np.array(kernels)

    return (kernels)

def _get_kernels(max_radius,linear_steps):

    if linear_steps>max_radius: linear_steps=max_radius

    vals = np.exp2(np.arange(1,np.ceil(np.log2(max_radius))))
    max_radius = int(vals[-1])
    # print(vals)
    # vals = np.ones((max_radius)+1)
    # vals[0]=0
    # vals[linear_steps+1:]+=np.arange(max_radius-linear_steps)
    # vals = np.cumsum(vals)
    # vals = vals[:(vals<max_radius).sum()+1]
    # vals[-1]=max_radius

    span = np.arange(-max_radius , max_radius +1)
    X,Y = np.meshgrid(span,span)
    dists = (X**2+Y**2)**0.5

    kernel_1 = np.zeros_like(X)
    kernel_1[max_radius,max_radius]=1
    kernels = [kernel_1]

    for i in range(len(vals)-1):

        r1 = vals[i]
        r2 = vals[i]+1
        
        kernel_1 = (dists-r1)
        kernel_1 = -(kernel_1-1)*(kernel_1<1)
        kernel_1[kernel_1>1] = 1
        kernel_1 = 1-kernel_1 

        kernel_2 = (dists-r2)
        kernel_2 = -(kernel_2-1)*(kernel_2<1)
        kernel_2[kernel_2>1] = 1
        kernel_2[kernel_2==1] = kernel_1[kernel_2==1]
        
        kernels.append(kernel_2)
        
    kernels = np.array(kernels)

    return (kernels)

    
class Ripley():
    """Ripley class 

    Attributes
    ----------
    radius : int
        Total radius of the kernel in pixels 
    resolution : int
        The step size / resolution of the algorithm in pixels 

    Methods
    -------

    """

    def __init__(self, um_per_px=None):
        
        self.radius = None#radius
        self.radii = None
        self.n_structures = None
        self.n_steps = None
        self.resolution = None#resolution
        self.kernels = None #_get_kernels(radius,resolution)
        self.kernel_areas = None
        self.tissue_mask = None
        self.um_per_px=um_per_px

        self.cell_map=None
        self.cell_matrix = None         
        self.coocurrences = None
        self.entropy = None
        self.total_probs = None
        self.correlogram=None

    def compute_coocurrences(self, celltype_map=None, max_radius=None, linear_steps=10, 
                            cell_matrix=None, edge_correction=False, tissue_mask=None, disable_progbar=True):
        '''Find surrounding cell occurrences in a cell matrix

        Attributes
        ----------  

        cell_matrix: list, nd_array
            a list of cell-wise entries of boolean 2d maps of spatial cell occurrence

        '''

        self._preprocess_map(celltype_map, max_radius, linear_steps, cell_matrix, edge_correction)

        img_shape = self.cell_matrix[0].shape
        kernel_shape = self.kernels[0].shape
        padded_shape = [kernel_shape[i]+img_shape[i] for i in range(2)]
        slide = kernel_shape[0]//2
        current_kernel = torch.zeros(padded_shape, device=device, dtype=torch.float) 
        current_sample = torch.zeros(padded_shape, device=device, dtype=torch.float)

        coocurrences = [[] for s in range(len(cell_matrix))]
        stack_ffts = []
        if tissue_mask is None: 
            if self.tissue_mask is None:
                self.tissue_mask = torch.ones(img_shape,dtype=np.bool)
            tissue_mask = self.tissue_mask
        self.tissue_mask_mapped=[]

        for s in tqdm_notebook(range(len(cell_matrix)), disable=disable_progbar):
            current_sample[:img_shape[0],:img_shape[1]] = torch.tensor(cell_matrix[s].astype(np.float), device=device)
            stack_ffts.append(torch.fft.rfft2(current_sample))
            
        # print(np.array(self.kernels).shape)

        for m,kernel in tqdm_notebook(enumerate(self.kernels),total=len(self.kernels), disable=disable_progbar):
            current_kernel[:kernel_shape[0],:kernel_shape[1]] = torch.tensor(kernel.astype(np.float), device=device)
            current_kernel_fft = torch.fft.rfft2(current_kernel)
                        
            for s in range(len(cell_matrix)):

                prod_gpu = (current_kernel_fft * stack_ffts[s])
                inv_gpu = torch.fft.irfft2(prod_gpu)                
                coocurrences[s].append(torch.abs(inv_gpu[slide:img_shape[0]+slide,
                                                    slide:img_shape[1]+slide])[tissue_mask].cpu().numpy().squeeze())

            if tissue_mask is not None:
                current_sample[:img_shape[0],:img_shape[1]] = torch.tensor(tissue_mask, dtype=torch.float, device=device)
                kernel_fft =  torch.fft.rfft2(current_sample)
                prod_gpu = (current_kernel_fft * kernel_fft)
                inv_gpu = torch.fft.irfft2(prod_gpu)      
                self.tissue_mask_mapped.append(torch.abs(inv_gpu[slide:img_shape[0]+slide,
                                                    slide:img_shape[1]+slide]).cpu().numpy().squeeze())

        self.coocurrences = np.array(coocurrences)

        if tissue_mask is not None and edge_correction:
            if type(    tissue_mask)==np.ndarray:
                tissue_mask = torch.tensor(tissue_mask)
            tissue_mask = tissue_mask.cpu()
            self.coocurrences = self.coocurrences/(0.01+np.array(self.tissue_mask_mapped)[:,tissue_mask])

        self.total_probs=self.cell_matrix.float().mean(-1).mean(-1) 

        return self.coocurrences

    def _preprocess_map(self, celltype_map, max_radius, linear_steps, cell_matrix, edge_correction):

        if max_radius is None:
            if celltype_map is None and cell_matrix is not None:
                max_radius = sum(cell_matrix.shape[1:])//8
            else:
                max_radius = sum(celltype_map.shape)//8
            print('No radius provided. Inferred '+str(max_radius)+' from cell map shape')

        self.radius=max_radius
        self.kernels = _get_kernels(max_radius,linear_steps)
        self.n_steps=len(self.kernels)
        self.kernel_areas = np.sum(self.kernels,-1).sum(-1)

        if cell_matrix is None:
            if celltype_map is None:
                print('Please provide either a cell matrix or a celltype map.')
            else:
                cell_matrix = torch.tensor([celltype_map.squeeze()==i for i in range(celltype_map.max()+1)],device=device)
        else:
            cell_matrix=torch.tensor(cell_matrix,device=device)
            if celltype_map is not None:
                print('Got both cell matrix and cell map input. Reverting to cell matrix.')

        if self.tissue_mask is not None:
            cell_matrix[:,~self.tissue_mask]=0

        if type(cell_matrix)==np.ndarray:
            self.cell_matrix = torch.tensor(cell_matrix,device=device)
        else:
            self.cell_matrix = cell_matrix.to(device)

    def rescale(self, matrix, factor):
        if self.um_per_px is not None:
            self.um_per_px/=factor
        return (transform.rescale(matrix,[1,factor,factor]))

    def radial_integration(self, cell_matrix=None, coocurrences=None, disable_progbar=True):
        '''

        '''

        if cell_matrix is not None: self.cell_matrix=cell_matrix
        if coocurrences is not None: self.coocurrences=coocurrences 

        cell_matrix_cpu = self.cell_matrix[:,self.tissue_mask].cpu()

        self.correlogram = []

        for m in tqdm_notebook(range(self.coocurrences.shape[1]),disable=disable_progbar):
            am=self.coocurrences[:,m]
            self.correlogram.append(np.inner(am,cell_matrix_cpu))
        # print(self.tissue_mask.shape)
        self.correlogram = np.transpose(self.correlogram,[2,1,0])[:,:]

        self.autocorrelations = self.correlogram.diagonal().T

        return self.correlogram

    def get_entropy(self):
        '''

        '''
        if cell_matrix is None:
            cell_matrix = self.coocurrences

    def _create_filter(self, radius, min_cells):
        '''Uses a circular convolution method to determinde the edges of tissue.
        '''

        #Create circular mesh:
        ftr = np.zeros([2*radius+1]*2) 
        span = np.linspace(-radius,radius,2*radius+1)
        X,Y = np.meshgrid(span,span)
        Z = (X**2+Y**2)**0.5<=radius

        #Create half slices:
        Zs_straight = []
        Zs_skewed = []
        for sign in [-1,1]:
            Zs_straight.append(Z*((((X+Y)<0)*((sign*X+(-sign)*Y)<0))*((-X-Y)<0)+((sign*X+(-sign)*Y)<0)))
            Zs_straight.append(Z*((((X+Y)<0)*((sign*X+(sign)*Y)>0))*((-X-Y)<0)+((sign*X+(sign)*Y)>0)))
            Zs_skewed.append(Z*((sign*X)>0))
            Zs_skewed.append(Z*((sign*Y)>0))

        #Overlay to create quarter slices:
        Zs_final = []
        for i in range(4):
            Zs_final.append(Zs_straight[i]*Zs_straight[(i+1)%len(Zs_straight)])
            Zs_final.append(Zs_skewed[i]*Zs_skewed[(i+1)%len(Zs_skewed)])

        return Zs_final

    def create_tissue_mask(self, cell_map=None, radius=None, min_cells=None):
        '''Creates tissue kernel from _create_filter() output.
        '''
        
        if radius is None:
            radius = (cell_map.sum()**0.5/8).astype(np.int)
            print('No filter radius provided. Inferred '+str(radius)+' from cell map shape')

        if min_cells is None:
            min_cells = (cell_map.sum()/4000)#.astype(np.int)
            print('No filter threshold provided. Inferred '+str(min_cells)+' from cell map content')

        filters = self._create_filter(radius, min_cells)

        t_map = torch.tensor(cell_map, dtype=torch.float, device=device)
        map_freqs = torch.fft.rfft2(t_map)#,2)

        t_filter = torch.zeros_like(t_map, device=device)
        t_sum = torch.zeros_like(t_map, device=device)

        for ftr in filters:
            t_filter[:ftr.shape[0],:ftr.shape[1]] = torch.tensor(ftr.astype(np.float), device=device)
            filter_freqs = torch.fft.rfft2(t_filter)#,dim=2)
            prod_gpu = (map_freqs * filter_freqs)
            inv_gpu = torch.fft.irfft2(prod_gpu)#,2)
            t_sum[:inv_gpu.shape[0],:inv_gpu.shape[1]] += inv_gpu>min_cells


        self.tissue_mask = torch.roll(torch.roll(t_sum,-int(radius),0),-int(radius),1)>=7 

        return self.tissue_mask

    def run_pipeline(self, max_radius, cell_map=None, linear_steps=None, filter_radius=None, mask=False,
                    filter_threshold=None ,plot=True, edge_correction=False, cell_matrix=None, flat_tail=False, disable_progbar=False):

        if linear_steps is None or linear_steps>max_radius : linear_steps=max_radius

        if cell_matrix is not None:
            cell_contours = cell_matrix.sum(0)>0
        else:
            cell_contours = cell_map

        if mask:
            print('Creating tissue kernels...')
            mask = self.create_tissue_mask(cell_contours, filter_radius, filter_threshold)
            print('Using %f percent of image surface'%mask.cpu().numpy().mean())
        else:
            mask = torch.ones(cell_contours.shape,dtype=np.bool)
            self.tissue_mask=mask

        print('Creating spatial coo-occurence maps...')
        am = self.compute_coocurrences(cell_map,max_radius,linear_steps,edge_correction=edge_correction,cell_matrix=cell_matrix,disable_progbar=disable_progbar)

        print('Determining correlations...')
        dm = self.radial_integration(disable_progbar=disable_progbar)

        if plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=[25,15])

            plt.subplot(2,3,1)
            plt.title('mask:')
            plt.imshow(mask.cpu(),aspect='auto')
            plt.subplot(2,3,2)
            plt.title('mask over tissue:')
            plt.imshow(torch.tensor(mask).cpu()-(2*(cell_contours>0)),aspect='auto')

            plt.subplot(2,3,3)
            plt.title('Celltype 0 ('+str(self.radius)+' px):')
            plt.imshow(am[0],aspect='auto')
            plt.subplot(2,3,4)
            plt.title('Celltype -1, lowest radius:')
            plt.imshow(am[-1],aspect='auto')

            plt.subplot(2,3,5)
            img = (self.autocorrelations[:,1:]-self.autocorrelations.min(1)[:,np.newaxis])
            img = (img/img.max(1)[:,np.newaxis])[:,1:]
            plt.title('Autocorrelation plot:')
            plt.imshow(img,aspect='auto')

            plt.subplot(2,3,6)
            plt.imshow(np.array(self.kernels)[:,(self.kernels[0].shape[0])//2],aspect='auto')

    def plot_distances(self, labels=None, scale=True, include_self=False):

        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        shp = self.correlogram.shape
        dm  = self.correlogram
        if self.um_per_px is None:
            um_per_px=1
            xlabel="distance(px)"
        else:
            um_per_px=self.um_per_px
            xlabel="distance($\mu m$)"

        xlox = np.linspace(0,self.n_steps-1,10)
        xtix = xlox*um_per_px*self.radius/self.n_steps

        fig = plt.figure(figsize=[15,shp[0]*3])

        spec = gridspec.GridSpec(ncols=2, nrows=shp[0]//2+2, figure=fig)

        ax = fig.add_subplot(spec[0,0:])


        img = (self.autocorrelations[:,1:])#-rip.autocorrelations.min(1)[:,np.newaxis])
        if scale: img = (img/img.max(1)[:,np.newaxis])[:,1:]
        plt.title('Autocorrelation plot:')
        ax.imshow(img,aspect='auto')
        plt.yticks(np.arange(shp[1]),labels)
        plt.xticks(xlox,["%.2f" %x for x in xtix])
        plt.xlabel('distance $(\mu m)$')

        for i in range(shp[0]):
            idcs = [s for s in range(shp[1]) if s!=i]
            
            xpos = ((i//2)+1)
            ypos = i%2
            
            ax = fig.add_subplot(spec[xpos,ypos])
            plt.title(labels[i])
                
            if scale:
                img = self.correlogram[i]**2/(self.correlogram[i]**2).min(1)[:,np.newaxis]
                img = (img/img.max(1)[:,np.newaxis])[:,1:]
            else:
                img=self.correlogram[i,1:]
            
            if include_self:
                ax.imshow(img,aspect='auto',cmap='viridis')
                plt.yticks(np.arange(shp[1]),labels)

            else:
                ax.imshow(img[idcs],aspect='auto',cmap='viridis')
                plt.yticks(np.arange(shp[1]-1),[labels[i] for i in idcs])
            plt.xticks(xlox,["%.2f" %x for x in xtix])
            

        plt.tight_layout()

    def plot_self_information(self):
        for i in range(len(categories)):

            plot = ((dm_cat[i,i]/dm_cat[i].sum(0))/(counts_cat[i]/counts_cat.sum())) # I'm a vertex pixel. 
            plot[plot>3]=np.nan
            _=plt.plot(plot,c=clrs[i])

    def determine_radii(self, cutoff = 0.32):

        n_structures = []
        radii = []

        for v in range(self.correlogram.shape[0]):
            
            plot = self.correlogram[v,v].copy()
            plot = plot/self.kernel_areas
            h=plot[0]
            r = np.where(plot<(h*cutoff))[0]
            if len(r): r=r[0]
            else: r=len(plot)

            radii.append(r)
            n_structures.append(h/r**2)
        
        self.radii = radii
        self.n_structures = n_structures

        return(n_structures,radii)        

    def statistical_analysis_old(self, cutoff = 3.9):

        maps = self.correlogram

        a_total =  maps[-1,-1,0] # cell_matrix[-1].sum()#
        a_cat = maps[np.arange(maps.shape[0]),-1,0]#cell_matrix.sum(-1).sum(-1)#

        n_structures,radii = self.determine_radii(cutoff=cutoff)

        sigs_low = np.zeros((maps.shape[0],maps.shape[0],maps.shape[-1],maps.shape[-1],))#[]
        sigs_high = np.zeros((maps.shape[0],maps.shape[0],maps.shape[-1],maps.shape[-1],))#[]
        # obs_exp = np.zeros((maps.shape[0],maps.shape[0],maps.shape[-1],maps.shape[-1],))#[]
        # obs_fact = np.zeros((maps.shape[0],maps.shape[0],maps.shape[-1],maps.shape[-1],))#[]
        descriptions = []

        for v in range(maps.shape[0]):

            descriptions.append([])
            for i in range(maps.shape[0]):
                
                a_i = a_cat[i]     # total area of 'i'
                a_v = a_cat[v]     # total area of 'v'  
                
                p_i = a_i/a_total  # probability of observing 'i' at any point in the sample
                p_v = a_v/a_total  # probability of observing 'v' at any point in the sample
                        
                rad_v = radii[v] # expected radius (v)
                rad_i = radii[i] # expected radius (i)   

                surf_v = rad_v**2*np.pi # surface of a 'v' structure
                surf_i = rad_i**2*np.pi # surface of an 'i' structure
                # plots_low = []
                # plots_high = []                
                dmmt =  maps[v,-1].sum()/a_total*a_i/maps[v,i].sum()  #2b reasoned for!
                
                for s in range(1,maps.shape[-1]):
                    
                    expected_observations = np.convolve(maps[v,-1],np.ones((s,)),mode='valid')/a_v/surf_i*a_i/a_total  # 2b reasoned for!
                    factual_observations  = np.convolve(maps[v,i],np.ones((s,)),mode='valid')/a_v/surf_i*dmmt
                    
                    sig_low = gammaincc(factual_observations,expected_observations, )
                    sig_high = gammaincc(factual_observations+1,expected_observations, )

                    sig_low[np.isnan(sig_low)]=0
                    sig_low[np.isinf(sig_low)]=0 
                    sig_high[np.isnan(sig_high)]=0
                    sig_high[np.isinf(sig_high)]=0        

                    sigs_low[i,v,s] = (scipy.signal.resample((sig_low),maps.shape[-1]))
                    sigs_high[i,v,s] = (scipy.signal.resample((sig_high),maps.shape[-1]))

        return (sigs_low, sigs_high)

    def statistical_analysis(self, cutoff = 3.9, disable_progbar=False):

        def intersection(d,r1,r2):
            return r1**2*np.arccos((d**2+r1**2-r2**2)/(2*d*r1))+ \
            r2**2*np.arccos((d**2+r2**2-r1**2)/(2*d*r2))- \
            1/2*np.sqrt((-d+r2+r1)*(d+r2-r1)*(d-r2+r1)*(d+r2+r1))

        resolution = 50

        def get_dist(s,d,theta):
            a = np.sin(theta)*s
            b = (d-np.cos(theta)*s)
            return (a**2+b**2)**0.5

        map_shape = self.correlogram.shape

        a_total =  self.correlogram[-1,-1,0] # cell_matrix[-1].sum()#

        surfaces = self.correlogram[-1,:,0] 
        probabilities = surfaces/surfaces[-1]

        n_structures,radii = self.determine_radii(cutoff=cutoff)
        radii = np.array(radii)
        exp_counts = surfaces/(radii**2*np.pi)

        sigs_low = np.zeros((map_shape[0],map_shape[0],map_shape[-1],map_shape[-1],))#[]
        sigs_high = np.zeros((map_shape[0],map_shape[0],map_shape[-1],map_shape[-1],))#[]
        
        descriptions = []

        d = np.arange(map_shape[-1])
        s = np.arange(map_shape[-1])
        theta = np.linspace(0,np.pi,resolution)

        S,D,T = np.meshgrid(s,d,theta)

        for v in tqdm_notebook(range(map_shape[0]), disable=disable_progbar):

            descriptions.append([])
            for i in range(map_shape[0]):

                r1 = radii[v]
                r2 = radii[i]
                    
                with np.errstate(divide='ignore'):
                    with np.errstate(invalid='ignore'):
                        ds = get_dist(S,D,T)
                        area = intersection(ds,r2,r1)
                        
                r_s = r1 if r1<r2 else r2
                inside = np.isnan(area)*(ds<r2)
                area[inside]=r_s**2*np.pi
                circle_correction = np.nansum(area,-1)/resolution
                circle_correction = circle_correction.max(0)

                circle_correction = circle_correction/circle_correction[-1]

                exp = (self.correlogram[v,-1]*probabilities[i]/circle_correction/surfaces[v]*exp_counts[v])
                obs = (self.correlogram[v,i]/circle_correction/surfaces[v]*exp_counts[v])

                for s in range(1,map_shape[-1],40):
                    
                    expected_observations = np.convolve(exp,np.ones((s,)),mode='valid')
                    factual_observations  = np.convolve(obs,np.ones((s,)),mode='valid')
                    
                    
                    sig_low = gammaincc(factual_observations,expected_observations, )
                    sig_high = gammaincc(factual_observations+1,expected_observations, )

                    sig_low[np.isnan(sig_low)]=0
                    sig_low[np.isinf(sig_low)]=0 
                    sig_high[np.isnan(sig_high)]=0
                    sig_high[np.isinf(sig_high)]=0        

                    sigs_low[i,v,s] = (scipy.signal.resample((sig_low),map_shape[-1]))
                    sigs_high[i,v,s] = (scipy.signal.resample((sig_high),map_shape[-1]))

        return (sigs_low, sigs_high)

    def save(self, path):
        outfile = TemporaryFile()

        # [print(i) for i in (path, self.radius, self.n_steps, self.resolution,
        # self.kernels,self.tissue_mask.cpu(),self.kernel_areas,  self.um_per_px,     
        # self.correlogram,  self.entropy, self.total_probs.cpu())]

        np.savez(path, self.radius, self.n_steps, self.resolution,
        self.kernels,self.tissue_mask.cpu(),self.kernel_areas,  self.um_per_px,     
        self.correlogram,  self.entropy, self.total_probs.cpu(), self.coocurrences)

    def load(self, path):

        npzfile = np.load(path,allow_pickle=True)

        ( self.radius, self.n_steps, self.resolution,
        self.kernels,self.tissue_mask,self.kernel_areas,  self.um_per_px,     
        self.correlogram,  self.entropy, self.total_probs, self.coocurrences) = [npzfile[f] for i,f in enumerate(npzfile.files)]

        npzfile.close()

    def load_and_integrate(self,paths):
# 
        npzfile = np.load(paths[0],allow_pickle=True)

        ( self.radius, self.n_steps, self.resolution,
        self.kernels,self.tissue_mask,self.kernel_areas,  self.um_per_px,     
        self.correlogram,  self.entropy, self.total_probs) = [npzfile[f] for i,f in enumerate(npzfile.files)]

        npzfile.close()

        if len(paths)>1:
            for path in paths:
                npzfile = np.load(path,allow_pickle=True)

                print(path)

                (radius, n_steps, resolution, kernels, tissue_mask, kernel_areas,
                um_per_px,   correlogram,  entropy, total_probs) = [npzfile[f] for i,f in enumerate(npzfile.files)]

                npzfile.close()

                if all([self.correlogram.shape[i]==s for i,s in enumerate(correlogram.shape)]):    
                    self.correlogram+=correlogram
                else:
                    print('Skipping %s, shapes not compatible...'%path)

