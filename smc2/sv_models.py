# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:20:08 2016

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv




import os
from smc2.Model import Model




class SVoneSP500Model(Model):
    def __init__(self):   
        super(Model, self).__init__()
        import userfiles.SMC2SVoneSP500 as mdl_param
        import models.SVonefactorx as mod_latent
        import models.SVonefactortheta as mod_theta
        
        
        data_path = os.path.split(__file__)[0]+'/../data/'+mdl_param.DATASET+'.R'        
        self.mdl_param = mdl_param
        self.latent = mod_latent.modelx
        self.latent.loadData(data_path)
        self.theta = mod_theta.modeltheta
        self.dim = 5

class SVmultiSP500Model(Model):
    def __init__(self):   
        super(Model, self).__init__()
        import userfiles.SMC2SVmultiSP500 as mdl_param
        import models.SVmultifactorx as mod_latent
        import models.SVmultifactortheta as mod_theta
        
        
        data_path = os.path.split(__file__)[0]+'/../data/'+mdl_param.DATASET+'.R'        
        self.mdl_param = mdl_param
        self.latent = mod_latent.modelx
        self.latent.loadData(data_path)
        self.theta = mod_theta.modeltheta
        self.dim = 9
