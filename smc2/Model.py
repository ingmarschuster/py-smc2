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

#untransform: apply exp/logit -> constrained space
#transform: apply log/inverse_logit -> unconstrained space

class Model(object):

    def __init__(self):
        pass
    
    def rvs(self, num_theta):
        '''
        Draw num_theta RVs from prior over theta in unconstrained space
        
        Returns
        =======
        generated_rvs   -  num_theta draws from the prior over theta
        '''
        
        #beware: Smc^2 assumes particles are columnvectors.
        #thus we have to transpose
        rval = self.theta.transform(self.theta.priorgenerator(num_theta)).T
        
        #the SMC^2 code comments say rval should contain untransformed (constrained) parameters
        #we transform them to get them unconstrained        
        
        return rval
  
    
    def logpdf(self, theta_particles):
        '''
        Get estimate of posterior densitz in logspace for all theta particles
        using a total of self.mdl_param.NX particles for the inner SMC that 
        estimates the likelihood
        
        Parameters
        ==========
        theta_particles   -  Particles for the parameters, assumed to be in the unconstrained (transformed) space 
        
        Returns
        =======
        log_estimate_posterior - estimate of the likelihood in log space
        '''
        
        
        #beware: Smc^2 assumes particles are columnvectors, and everything is at least 2d
        #thus we have to transpose
        theta_particles = np.atleast_2d(theta_particles).T
        
        constr_theta_particles = self.theta.untransform(theta_particles)
        assert(not(np.any(np.isnan(constr_theta_particles)) or np.any(np.isinf(constr_theta_particles))))
        
        inner_smc = ParallelSIRs(self.mdl_param.NX, constr_theta_particles, self.latent.model_obs, self.latent); inner_smc.first_step(); inner_smc.next_steps()
        

        lprior = self.theta.priorlogdensity(theta_particles) # Takes unconstrained (transformed) parameters
        
        return  lprior + inner_smc.getTotalLogLike() 
    
    def get_logpdf_closure(self):
        rval = lambda theta: self.logpdf(theta)
        return rval

        