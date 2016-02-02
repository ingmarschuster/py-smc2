###################################################
#    This file is part of py-smc2.
#    http://code.google.com/p/py-smc2/
#
#    py-smc2 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    py-smc2 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with py-smc2.  If not, see <http://www.gnu.org/licenses/>.
###################################################

#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import os, os.path, time
from numpy import random, power, sqrt, exp, zeros, \
        ones, mean, average, prod, log, sum, repeat, \
        array, float32, int32, cov, isnan, zeros_like, \
        var, isinf, linalg, pi, dot, argmax, transpose, diag, \
        newaxis, outer, minimum, triu_indices, apply_along_axis
from numpy import max as numpymax
from numpy import min as numpymin
from numpy import sum as numpysum
from scipy.stats import norm
from scipy.spatial import distance
from resampling import IndResample, resample2D
from various import ESSfunction, progressbar
from parallelSIRs import ParallelSIRs
import cPickle as pick

class SMCsquare:
    """
    """
    def __init__(self, model, algorithmparameters, \
            dynamicNx = True, savingtimes = [], autoinit = True, pickle = False):
        ## essential things
        # get the model:
        self.modelx = model["modelx"]
        self.modeltheta = model["modeltheta"]
        self.observations = model["observations"]
        self.statedimension = self.modelx.xdimension
        self.obsdimension = self.modelx.ydimension
        # get the basic algorithmic parameters
        self.AP = algorithmparameters
        self.Nx = algorithmparameters["Nx"]
        self.Ntheta = algorithmparameters["Ntheta"]
        self.T = self.observations.shape[0]
        self.excludedobservations = self.modelx.excludedobservations
        # initialize main matrices and vectors
        self.thetaparticles = zeros((self.modeltheta.parameterdimension, self.Ntheta))
        self.transformedthetaparticles = zeros((self.modeltheta.parameterdimension, self.Ntheta))
        self.thetalogweights = zeros((self.T, self.Ntheta))
        self.xparticles = zeros((self.Nx, self.statedimension, self.Ntheta))
        self.xweights = zeros((self.Nx, self.Ntheta))
        self.logxweights = zeros((self.Nx, self.Ntheta))
        self.constants = zeros(self.Ntheta)
        self.logLike = zeros(self.Ntheta)
        self.totalLogLike = zeros(self.Ntheta)
        self.priordensityeval = zeros(self.Ntheta)
        self.evidences = zeros(self.T)
        ## Filtering and Smoothing
        self.filtered = {}
        if self.AP["filtering"]:
            if hasattr(self.modelx, "filteringlist"):
                self.filtered = []
                for d in self.modelx.filteringlist:
                    self.filtered.append(zeros((self.T, d["dimension"])))
        self.smoothingEnable = algorithmparameters["smoothing"]
        self.smoothedmeans = {}
        self.smoothedvalues= {}
        if self.smoothingEnable:
            self.storesmoothingtime = algorithmparameters["storesmoothingtime"]
        ## Prediction
        if self.AP["prediction"]:
            if hasattr(self.modelx, "predictionlist"):
                self.predicted = []
                for d in self.modelx.predictionlist:
                    self.predicted.append(zeros((self.T, d["dimension"])))
        ## other things:
        # store ESS at each time
        self.ESS = zeros(self.T)
        # store iterations where resample-moves are performed
        self.resamplingindices = []
        # store the acceptance ratios of each move step
        self.acceptratios = []
        #self.guessacceptratios = []
        # store all Nx 
        # (in case it is automatically increasing along the iterations)
        self.Nxlist = [self.Nx]
        # store iterations at each Nx is increased
        self.increaseindices = [0]
        # iterations at which all the theta-particles are saved
        self.savingtimes = savingtimes
        self.savingtimes.append(self.T)
        # initialize matrices to store the weighted theta-particles
        self.thetahistory = zeros((len(self.savingtimes), self.modeltheta.parameterdimension, self.Ntheta))
        self.weighthistory = zeros((len(self.savingtimes), self.Ntheta))
        # number of already past saving times
        self.alreadystored = 0
        # computing time for each iteration
        self.computingtimes = zeros(self.T)
        print "------------------"
        print "launching SMC^2, with algorithm parameters:"
        for key, element in algorithmparameters.items():
            print key, ":", element
        print "------------------"
        if pickle:
            with open('model.pick', 'w') as f:
                pick.dump(self, f)
            return
        if autoinit:
            self.first_step()
            self.next_steps()

    def increaseParticlesNb(self, t):
        """
        Double the number of x-particles.
        """
        print "increasing Nx: from %i to %i" % (self.Nx, 2 * self.Nx)
        self.Nx = 2 * self.Nx
        biggerSIRs = ParallelSIRs(self.Nx, self.thetaparticles, self.observations[0:(t+1),:], self.modelx)
        biggerSIRs.first_step()
        biggerSIRs.next_steps()
        biggerTotalLogLike = biggerSIRs.getTotalLogLike()
        self.thetalogweights[t, :] = self.thetalogweights[t, :] +  biggerTotalLogLike - self.totalLogLike
        self.totalLogLike = biggerTotalLogLike.copy()
        self.xparticles = biggerSIRs.xparticles.copy()
        self.xweights = zeros_like(biggerSIRs.xweights)
        self.logxweights = zeros_like(biggerSIRs.xweights)
        self.Nxlist.append(self.Nx)
        self.increaseindices.append(t)

    def xresample(self):
        """
        Resample all the x-particles according to their weights.
        """
        self.xparticles[...] = resample2D(self.xparticles, self.xweights, self.Nx, self.statedimension, self.Ntheta)
    def thetaresample(self, t):
        """
        Resample the theta-particles according to their weights.
        """
        indices = IndResample(exp(self.thetalogweights[t, :]), self.Ntheta)
        self.thetaparticles[...] = self.thetaparticles[:, indices]
        self.transformedthetaparticles[...] = self.transformedthetaparticles[:, indices]
        self.xparticles[...] = self.xparticles[:, :, indices]
        self.totalLogLike[:] = self.totalLogLike[indices]
        self.priordensityeval[:] = self.priordensityeval[indices]
        self.thetalogweights[t, :] = 0.

    def PMCMCstep(self, t):
        """
        Perform a PMMH move step on each theta-particle.
        """
        
        #### INGMAR: This is where the proposal is computed
        transformedthetastar = self.modeltheta.proposal(self.transformedthetaparticles, \
                self.proposalcovmatrix, hyperparameters = self.modeltheta.hyperparameters,\
                proposalmean = self.proposalmean, proposalkernel = self.AP["proposalkernel"])
        thetastar = self.modeltheta.untransform(transformedthetastar)
        
        ###### INGMAR: This is where the likelihood is estimated
        proposedSIRs = ParallelSIRs(self.Nx, thetastar, self.observations[0:(t+1)], self.modelx)
        proposedSIRs.first_step()
        proposedSIRs.next_steps()
        proposedTotalLogLike = proposedSIRs.getTotalLogLike()
        
        
        acceptations = zeros(self.Ntheta)

        proposedpriordensityeval = apply_along_axis(func1d = self.modeltheta.priorlogdensity,\
                arr = transformedthetastar, axis = 0)
                
        proposedlogomega = proposedTotalLogLike + proposedpriordensityeval
        currentlogomega = self.totalLogLike + self.priordensityeval
        # if proposal kernel == "randomwalk", then nothing else needs to be computed
        # since the RW is symmetric. If the proposal kernel is independent, then
        # the density of the multivariate gaussian used to generate the proposals
        # needs to be taken into account
        if self.AP["proposalkernel"] == "randomwalk":
            pass
        elif self.AP["proposalkernel"] == "independent":
            invSigma = linalg.inv(self.proposalcovmatrix)
            def multinorm_logpdf(x):
                centeredx = (x - self.proposalmean)
                return -0.5 * dot(dot(centeredx, invSigma), centeredx)
            proposaltermstar = apply_along_axis(func1d = multinorm_logpdf, \
                    arr = transformedthetastar, axis = 0)
            proposedlogomega -= proposaltermstar
            proposaltermcurr = apply_along_axis(func1d = multinorm_logpdf, \
                    arr = self.transformedthetaparticles, axis = 0)
            currentlogomega  -= proposaltermcurr
        for i in range(self.Ntheta):
            acceptations[i]  = (log(random.uniform(size = 1)) < (proposedlogomega[i] - currentlogomega[i]))
            if acceptations[i]:
                self.transformedthetaparticles[:, i] = transformedthetastar[:, i].copy()
                self.thetaparticles[:, i] = thetastar[:, i].copy()
                self.totalLogLike[i] = proposedTotalLogLike[i].copy()
                self.priordensityeval[i] = proposedpriordensityeval[i].copy()
                self.xparticles[:, :, i] = proposedSIRs.xparticles[:, :, i].copy()
        acceptrate = sum(acceptations) / self.Ntheta
        self.acceptratios.append(acceptrate)
    def first_step(self):
        """
        First step: generate Ntheta theta-particles from the prior, and then
        for each theta_i, simulate Nx x-particles from the initial distribution
        p(x_0 | theta_i)
        """
        self.thetaparticles[...] = self.modeltheta.priorgenerator(self.Ntheta)
        self.transformedthetaparticles[...] = self.modeltheta.transform(self.thetaparticles)
        for i in range(self.Ntheta):
            self.xparticles[:, :, i] = self.modelx.firstStateGenerator(self.thetaparticles[:, i], size = self.Nx)
        self.priordensityeval = apply_along_axis(func1d = self.modeltheta.priorlogdensity,\
                arr = self.transformedthetaparticles, axis = 0)
    def next_steps(self):
        """
        Perform all the iterations until time T == number of observations.
        """
        for t in range(self.T):
            excluded = t in self.excludedobservations
            progressbar(t / (self.T - 1))
            if excluded:
                print "\nobservations", self.observations[t,:], "set to be excluded"
            last_tic = time.time()
            TandWresults = self.modelx.transitionAndWeight(self.xparticles, \
                    self.observations[t], self.thetaparticles, t + 1)
            self.xparticles[...] = TandWresults["states"]
            assert()
            if not(excluded):
                self.logxweights[...] = TandWresults["weights"]
                # in case the measure function returns nans or infs, set the weigths very low
                self.logxweights[isnan(self.logxweights)] = -(10**150)
                self.logxweights[isinf(self.logxweights)] = -(10**150)
                self.constants[:] = numpymax(self.logxweights, axis = 0)
                self.logxweights[...] -= self.constants[:]
            else:
                self.logxweights = zeros((self.Nx, self.Ntheta))
                self.constants[:] = numpymax(self.logxweights, axis = 0)
            self.xweights[...] = exp(self.logxweights)
            self.logLike[:] = log(mean(self.xweights, axis = 0)) + self.constants[:]
            # prediction: at this point we have the transitioned x-particles and we didn't update
            # the weights of the theta-particles, and the x-particles are not weighted
            if self.AP["prediction"]:
                self.prediction(t)
            if t > 0:
                self.evidences[t] = self.getEvidence(self.thetalogweights[t-1, :], self.logLike)
                self.totalLogLike[:] += self.logLike[:]
                self.thetalogweights[t, :] = self.thetalogweights[t-1, :] + self.logLike[:]
            else:
                self.evidences[t] = self.getEvidence(self.thetalogweights[t, :], self.logLike)
                self.totalLogLike[:] += self.logLike[:]
                self.thetalogweights[t, :] = self.thetalogweights[t, :] + self.logLike[:]
            self.thetalogweights[t, :] -= max(self.thetalogweights[t, :])
            self.xresample()
            self.ESS[t] = ESSfunction(exp(self.thetalogweights[t, :]))
            if self.AP["dynamicNx"]:
                progressbar(t / (self.T - 1), text = " ESS: %.3f, Nx: %i" % (self.ESS[t], self.Nx))
            else:
                progressbar(t / (self.T - 1), text = " ESS: %.3f" % self.ESS[t])
            while self.ESS[t] < (self.AP["ESSthreshold"] * self.Ntheta):
                progressbar(t / (self.T - 1), text =\
                        " ESS: %.3f - resample move step at iteration = %i" % (self.ESS[t], t))
                covdict = self.computeCovarianceAndMean(t)
                
                if self.AP["proposalkernel"] == "randomwalk":
                    self.proposalcovmatrix = self.AP["rwvariance"] * covdict["cov"]
                    self.proposalmean = None
                elif self.AP["proposalkernel"] == "independent":
                    self.proposalcovmatrix = covdict["cov"]
                    self.proposalmean = covdict["mean"]
                    
                self.thetaresample(t)
                self.resamplingindices.append(t)
                self.ESS[t] = ESSfunction(exp(self.thetalogweights[t, :]))
                
                ###### INGMAR: this is where PMCMC moves are made. we can do PMC proposals instead here
                for move in range(self.AP["nbmoves"]):
                    self.PMCMCstep(t)
                    acceptrate = self.acceptratios[-1]
                    progressbar(t / (self.T - 1), text = \
                            " \nresample move step at iteration = %i - acceptance rate: %.3f\n" % (t, acceptrate))
                    if self.acceptratios[-1] < self.AP["dynamicNxThreshold"] \
                            and self.Nx <= (self.AP["NxLimit"] / 2) \
                            and self.AP["dynamicNx"]:
                        self.increaseParticlesNb(t)
                        self.ESS[t] = ESSfunction(exp(self.thetalogweights[t, :]))
            new_tic = time.time()
            self.computingtimes[t] = new_tic - last_tic
            last_tic = new_tic
            """ filtering and smoothing """
            if self.AP["filtering"]:
                self.filtering(t)
            if self.smoothingEnable and t == self.T - 1:
                self.smoothing(t)
            if t in self.savingtimes or t == self.T - 1:
                print "\nsaving particles at time %i" % t
                self.thetahistory[self.alreadystored, ...] = self.thetaparticles.copy()
                self.weighthistory[self.alreadystored, ...] = exp(self.thetalogweights[t, :])
                self.alreadystored += 1
    def filtering(self, t):
        if hasattr(self.modelx, "filteringlist"):
            for index, d in enumerate(self.modelx.filteringlist):
                self.filtered[index][t, :] = d["function"](self.xparticles, \
                    exp(self.thetalogweights[t,:]), self.thetaparticles, t)
    def prediction(self, t):
        if hasattr(self.modelx, "predictionlist"):
            for index, d in enumerate(self.modelx.predictionlist):
                self.predicted[index][t, :] = d["function"](self.xparticles, \
                    exp(self.thetalogweights[t - 1,:]), self.thetaparticles, t)
    def smoothing(self, t):
        print "\n smoothing ..."
        ########
        from SIR import SIR
        smoothedx = zeros((t+1, self.statedimension, self.Ntheta))
        randomindices = random.randint(low = 0, high = self.Nx, size = self.Ntheta)
        newtotalLogLik = zeros(self.Ntheta)
        for indextheta in range(self.Ntheta):
            sir = SIR(self.Nx, self.thetaparticles[:, indextheta], \
                    self.observations[0:(t+1),:], self.modelx, storall = True)
            newtotalLogLik[indextheta] = sir.getTotalLogLike()
            onepath = sir.retrieveTrajectory(randomindices[indextheta])
            smoothedx[:, :, indextheta] = onepath[1:(t+2), :]
            #print smoothedx[:,:,indextheta]
            #raw_input()
        logw = self.thetalogweights[t, :] + newtotalLogLik - self.totalLogLike
        self.smoothedvalues["weights"] = zeros(self.Ntheta)
        self.smoothedvalues["weights"][:] = logw
        for key in self.modelx.smoothingfunctionals.keys():
            smoothkey = key + "%i" % (t + 1)
            self.smoothedmeans[smoothkey] = zeros(t + 1)
            self.smoothedvalues[smoothkey] = zeros(self.Ntheta)
            for subt in range(t + 1):
                smoothedxt = smoothedx[subt, ...]
                tempmatrix = self.modelx.smoothingfunctionals[key](smoothedx[newaxis, subt, ...], \
                                                                   self.thetaparticles, subt)
                tempmean = mean(tempmatrix, axis = 0)
                self.smoothedmeans[smoothkey][subt] = average(tempmean, weights = exp(logw))
                if subt == self.storesmoothingtime:
                    self.smoothedvalues[smoothkey][:] = tempmean
        print "...done!"

    def computeCovarianceAndMean(self, t):
        X = transpose(self.transformedthetaparticles)
        w = exp(self.thetalogweights[t, :])
        w = w / numpysum(w)
        weightedmean = average(X, weights = w, axis = 0)
        diagw = diag(w)
        part1 = dot(transpose(X), dot(diagw, X))
        Xtw = dot(transpose(X), w[:, newaxis])
        part2 = dot(Xtw, transpose(Xtw))
        numerator = part1 - part2
        denominator = 1 - numpysum(w**2)
        weightedcovariance = numerator / denominator
        # increase a little bit the diagonal to prevent degeneracy effects
        weightedcovariance += diag(zeros(self.modeltheta.parameterdimension) + 10**(-4)/self.modeltheta.parameterdimension)
        return {"mean": weightedmean, "cov": weightedcovariance}
    def getEvidence(self, thetalogweights, loglike):
        """
        Return the evidence at a given time
        """
        return average(exp(loglike), weights = exp(thetalogweights))
    def getResults(self):
        """
        Return a dictionary with vectors of interest.
        """
        resultsDict = {"Ntheta" : self.Ntheta, "Nx" : self.Nx, "T": self.T, \
                "thetahistory": self.thetahistory, "weighthistory": self.weighthistory, \
                "savingtimes": self.savingtimes, "ESS": self.ESS, "observations": self.observations, \
                "acceptratios": self.acceptratios, "Nxlist": self.Nxlist, \
                "increaseindices": self.increaseindices, "resamplingindices": self.resamplingindices, \
                "evidences": self.evidences, "smoothedmeans": self.smoothedmeans, \
                "smoothedvalues": self.smoothedvalues, \
                "computingtimes": self.computingtimes}
#        if self.AP["proposalkernel"] == "independent":
#            resultsDict.update({"guessAR": self.guessacceptratios})
        if self.AP["filtering"]:
            if hasattr(self, "filtered"):
                for index, d in enumerate(self.modelx.filteringlist):
                    resultsDict.update({"filtered%s" % d["name"]: self.filtered[index]})
        if self.AP["prediction"]:
            if hasattr(self, "predicted"):
                for index, d in enumerate(self.modelx.predictionlist):
                    resultsDict.update({"predicted%s" % d["name"]: self.predicted[index]})
        return resultsDict



