# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:11:39 2016

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv






from src.parallelSIRs import ParallelSIRs



__all__ = ['Model']


class Model(object):

    def __init__(self):
        pass
    
    def rvs(self, num_theta):
        '''
        Draw num_theta RVs from prior over theta
        Returns
        =======
        generated_rvs   -  num_theta draws from the prior over theta
        '''
        
        #beware: Smc^2 assumes particles are columnvectors.
        #thus we have to transpose
        return self.theta.priorgenerator(num_theta).T
    
    def get_ll_estimate(self, theta_particles):
        '''
        Get estimate of likelihood in logspace for all theta particles
        using a total of self.mdl_param.NX particles for the inner SMC that 
        estimates the likelihood
        
        Parameters
        ==========
        theta_particles   -  Particles for the parameters
        
        Returns
        =======
        log_estimate_likelihood - estimate of the likelihood in log space
        '''
        #beware: Smc^2 assumes particles are columnvectors.
        #thus we have to transpose
        inner_smc = ParallelSIRs(self.mdl_param.NX, theta_particles.T, self.latent.model_obs, self.latent)
        inner_smc.first_step()
        inner_smc.next_steps()
        return inner_smc.getTotalLogLike()
    
    def logpdf(self, theta_particles):
        '''
        Get estimate of posterior densitz in logspace for all theta particles
        using a total of self.mdl_param.NX particles for the inner SMC that 
        estimates the likelihood
        
        Parameters
        ==========
        theta_particles   -  Particles for the parameters
        
        Returns
        =======
        log_estimate_posterior - estimate of the likelihood in log space
        '''
        
        llhood = self.get_ll_estimate(theta_particles)
        
        #beware: Smc^2 assumes particles are columnvectors.
        #thus we have to transpose for self.theta.priorlogdensity
        lprior = self.theta.priorlogdensity(theta_particles.T) 
        
        return  lprior + llhood 

        