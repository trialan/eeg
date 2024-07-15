"""
Robinson Spectral-Domain Thalamocortical Neural Field Model Variants
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from numpy import pi,abs,exp,log,log10
from scipy import optimize

from ipywidgets import *

from copy import deepcopy

class Abeysuriya2015Model():
    
    def __init__(self):     
        
        # Constants
        self.gamma_e = 116 # s^-1
        self.r_e = 86 # mm
        self.Q_max = 340 # s^-1
        self.theta = 12.9 # mV
        self.sigma = 3.8 #mV
        self.phi_n = 10**-5 # s^-1
        self.k0 = 10
        
        self.G_rs = 0.1  # from Abeysuria 2015
        self.G_re = 0.2  # from Abeysuria 2015
        self.G_sn = 1. # Random <3 John
        
        self.l_x = self.l_y = 0.5
        
        self.fmax = 50
        self.freq_min = 5.
        self.freq_max = 100. 
        self.n_freqs = 50

        self.freqs = np.linspace(self.freq_min,self.freq_max,num=self.n_freqs)
        self.omega = 2. * pi * self.freqs

        # Variable parameters
        self.G_ee = 5.4
        self.G_ei = -7.
        self.G_ese = 100. # 5.6 # = G_es * G_se
        self.G_esre = -100. # -2.8 # = G_se * G_sr * G_re
        self.G_srs = -1. # -0.6 # = G_sr * G_rs
        
        self.alpha = 75 #s^-1
        self.beta = 75*3.8 #s^-1
        self.t0 = 84 # ms
        self.A_EMG = 0.5E-12 #s^-1
        self.f_EMG = 40 # Hz
        
        # Variable bounds
        self.bound_G_ee = [0., 20.]
        self.bound_G_ei = [-40., 0.]
        self.bound_G_ese = [0.,200] #[0., 40.]
        self.bound_G_esre = [-100., 100.] #[-40., 0.]
        self.bound_G_srs = [-20., 20] # [-14., -0.1]
        
        self.bound_alpha = [10., 400.] # 200.]
        self.bound_beta = [50., 1200.] # 800.]
        self.bound_t0 = [50., 140.]
        self.bound_A_EMG = [0.00001, 0.1] # 1E-12]
        self.bound_f_EMG = [10., 50.]
       

        # Optimization stuff
        self.optimize_train = None
        self.optimize_freqs = None
        self.optimize_param_list = None
        self.optimize_tol = None
        self.optimize_output = None
        
    def compute_L(self, omega):
        
        alpha, beta = self.alpha, self.beta
        L = (1 - 1j*omega/alpha)**-1 * (1 - 1j*omega/beta)**-1
        
        return L
    
    
    def compute_q2r2(self, omega):
        
        gamma_e = self.gamma_e
        G_ei, G_ee = self.G_ei, self.G_ee
        G_ese, G_esre, G_srs = self.G_ese, self.G_esre, self.G_srs
        t0 = self.t0
        
        L = self.compute_L(omega)
        
        term1 = (1 - 1j*omega / gamma_e)**2
        coeff2 = (1 - G_ei * L)**-1
        
        term2_1 = L * G_ee 
        term2_2 = (L**2 * G_ese + L**3 * G_esre) * exp(1j*omega*t0) / (1 - L**2 * G_srs)
        term2 = term2_1 + term2_2
        
        q2r2 = term1 - coeff2 * term2
        
        return q2r2
    
    
    def compute_k2r2(self, m, n):
        
        k_x = 2*pi*m / self.l_x
        k_y = 2*pi*n / self.l_y
        
        k2r2 = (k_x**2 + k_y**2)*self.r_e**2
        
        return k2r2
    
    
    def compute_P_EEG(self, freqs=None):
        
        if freqs is None: freqs = self.freqs

        omega = 2. * pi * freqs
        
        G_ei, G_ee = self.G_ei, self.G_ee
        G_ese, G_esre, G_srs = self.G_ese, self.G_esre, self.G_srs
        t0 = self.t0
        r_e = self.r_e
        k0 = self.k0
        
        phi_n = self.phi_n
        
        # Other Gs
        G_sr = G_srs / self.G_rs
        G_es = G_esre / (G_sr * self.G_re)
        G_sn = self.G_sn
        
        L = self.compute_L(omega)
        q2r2 = self.compute_q2r2(omega)
        
        term1 = G_es * G_sn * phi_n * L**2 * exp(1j*omega*t0/2)
        term2 = (1 - G_srs * L**2) * (1 - G_ei * L)
        
        term3 = 0
        k_x = 2 * pi / self.l_x
        k_y = 2 * pi / self.l_y
        fmax = self.fmax
        for m in np.arange(-fmax,fmax):
            for n in  np.arange(-fmax,fmax):
                
                k2r2 = self.compute_k2r2(m,n)
                k2 = k2r2 / r_e
                Fk = exp(-k2 / k0**2)
                term3 += abs(k2r2 + q2r2)**-2 * Fk * k_x * k_y
        
        P_EEG = abs(term1)**2 * abs(term2)**2 * term3 
  
    
        df_P_EEG = pd.DataFrame(P_EEG,index=freqs,columns=['power'])
        df_P_EEG.index.names = ['Hz']

        return P_EEG
    
    
    def compute_P(self, freqs,return_df=False,normalize=False):
        '''
        Compute the power spectrum.
        '''
        
        omega = 2. * pi * freqs

        A_EMG, f_EMG = self.A_EMG, self.f_EMG
        
        mod_omega = omega / (2 * pi * f_EMG)
        P_EMG = A_EMG * (mod_omega)**2 / (1 + mod_omega**2)**2
        
        P_EEG = self.compute_P_EEG(freqs)
        
        
        P_EEG_EMG = P_EEG + P_EMG
        
        if normalize: 
            P_EEG = P_EEG/P_EEG.max()
            P_EMG = P_EMG/P_EMG.max()
            P_EEG_EMG = P_EEG_EMG/P_EEG_EMG.max()
            
            P_EEG[np.isnan(P_EEG)] = 0
            P_EMG[np.isnan(P_EMG)] = 0
            P_EEG_EMG[np.isnan(P_EEG_EMG)] = 0
                
        if return_df == False:
            return P_EEG_EMG
        else:
            df = pd.DataFrame([P_EEG, P_EMG, P_EEG_EMG], columns=freqs,index=['P_EEG', 'P_EMG','P_EEG_EMG']).T
            df.index.names = ['Hz']
            return df
        
    
    
    def update_and_compute_P(self, values, param_list, omega,normalize=False):
        
        N = min(len(values), len(param_list))
        for k in range(N):
            setattr(self, param_list[k], values[k])
        
        P =  self.compute_P(omega,normalize=normalize)     
      
        return P


    def fit(self,data,freqs,param_list,tol,normalize=False,fit_log=False):
        '''
        Optimizing the Rowe Model onto a training set. The key parameters to adjust
        are as follows:
        - G_ee
        - G_ei
        - G_ese
        - G_esre
        - G_srs
        - alpha
        - beta
        - t0
        - A_EMG
        - f_EMG
        '''
    
        #self.optimize_train = train
        
        # Get frequencies
        #self.optimize_freqs = np.array([train[k][0] for k in range(len(train))])
        #self.optimize_output = np.array([train[k][1] for k in range(len(train))
        
        self.orig_mod = deepcopy(self)
       
        freqs = freqs.copy()
 
        omega = 2. * pi * freqs

        data = data.copy()

        
        if fit_log: data = np.log1p(data)

        if normalize:  data = data/data.max()

        data[np.isnan(data)] = 0
        
        self.optimize_freqs = freqs
        self.optimize_data = data
    
        '''
        Fits the model using the listed parameters.
        '''
        
        # Define the function w.r.t. the parameters. The vector P has the same
        # length as params, with 1-1 coordinate correspondance.
        
        if fit_log: 
          EEG_fun = lambda P: np.log1p(self.update_and_compute_P(P, param_list,freqs,normalize=normalize))
        else:
          EEG_fun = lambda P: self.update_and_compute_P(P, param_list,freqs,normalize=normalize)  
        chi_fun = lambda P: sum(((EEG_fun(P) - data) / data)**2)
        
        # Get initial parameter values
        P0 = []
        for j in range(len(param_list)):
            P0.append(getattr(self, param_list[j]))
        
        P0 = np.array(P0)
    
        # Obtain the bounds for the optimization procedure w.r.t. the selected
        # parameters.
        bounds_list = []
        for k in range(len(param_list)):
            
            bound_attr_str = 'bound_' + param_list[k]
            # Check if model has the bound attribute.
            if not hasattr(self, bound_attr_str):
                bounds_list.append((None,None))
            
            else:
                bounds_list.append(tuple(getattr(self, bound_attr_str)))
        
        bounds_tuple = tuple(bounds_list)
        
        # Initiate the optimization
        fit_result = optimize.minimize(chi_fun, P0, bounds=bounds_list, tol=tol)
        
        #  Update the params (seem to need to do this again...)
        for k,v in zip(param_list,fit_result['x']): setattr(self,k,v)
        
        
        fit_df = self.compute_P(freqs,return_df=True,normalize=normalize)
        fit_df['data'] = data
        
        
        return fit_result,fit_df
    

    def plot_widget(self,normalize=False,linestyle='-',logx=True,logy=True,xrange=[5,120],yrange=None):
        
        
        x = self.compute_P(self.freqs,normalize=normalize)
        
        
        self.widg_fig = plt.figure()
        self.widg_ax = self.widg_fig.add_subplot(1, 1, 1)
        self.widg_line, = self.widg_ax.plot(self.freqs,x,linestyle=linestyle) #x, np.sin(x))

        self.widg_ax.set_xlim(xrange) # 0,100])
        
        if normalize == False: 
            self.widg_ax.set_ylim([10E-10, 10E-1]) # 0,100])
        elif yrange != None:
            self.widg_ax.set_ylim(yrange)

        self.G_sr = self.G_srs / self.G_rs
        self.G_es = self.G_esre / (self.G_sr * self.G_re)
        
        
            
        self.widg_norm = normalize
            
        if logx: self.widg_ax.semilogx()
        if logy: self.widg_ax.semilogy()

   
        interact(self.update_widget,continuous_update=False,
                 G_ee=widgets.FloatSlider(min=self.bound_G_ee[0],max=self.bound_G_ee[1],step=1,value=self.G_ee),
                 G_ei=widgets.FloatSlider(min=self.bound_G_ei[0],max=self.bound_G_ei[1],step=1,value=self.G_ei),
                 G_ese=widgets.FloatSlider(min=self.bound_G_ese[0],max=self.bound_G_ese[1],step=1,value=self.G_ese),
                 G_esre=widgets.FloatSlider(min=self.bound_G_esre[0],max=self.bound_G_esre[1],step=1,value=self.G_esre),
                 G_srs=widgets.FloatSlider(min=self.bound_G_srs[0],max=self.bound_G_srs[1],step=1,value=-1.1), # 0.5), # self.G_srs),
                 alpha=widgets.FloatSlider(min=self.bound_alpha[0],max=self.bound_alpha[1],step=1,value=self.alpha),
                 beta=widgets.FloatSlider(min=self.bound_beta[0],max=self.bound_beta[1],step=1,value=self.beta),
                 t0=widgets.FloatSlider(min=self.bound_t0[0],max=self.bound_t0[1],step=1,value=self.t0),
                 A_EMG=widgets.FloatSlider(min=self.bound_A_EMG[0],max=self.bound_A_EMG[1],value=self.A_EMG,step=0.001),#step=1
                 f_EMG=widgets.FloatSlider(min=self.bound_f_EMG[0],max=self.bound_f_EMG[1],step=1,value=self.f_EMG))
                 
                 
                                        
                                        
    def update_widget(self,G_ee = 5.4,
                          G_ei = -7.,
                          G_ese = 5.6,
                          G_esre = -2.8,
                          G_srs = -0.6,
                          alpha = 75, #s^-1
                          beta = 75*3.8 ,
                          t0 = 84, # ms
                          A_EMG = 0.001, # 0.5E-12, #s^-1
                          f_EMG = 40):
            
            # Variable parameters
            self.G_ee = G_ee
            self.G_ei = G_ei
            self.G_ese = G_ese
            self.G_esre = G_esre
            self.G_srs = G_srs
        
            self.alpha = alpha
            self.beta = beta
            self.t0 = t0
            self.A_EMG = A_EMG
            self.f_EMG = f_EMG
            
            
            x = self.compute_P(self.freqs,normalize=self.widg_norm)
            
            self.widg_line.set_ydata(x)# np.sin(w * x))
            self.widg_fig.canvas.draw()
            

            
