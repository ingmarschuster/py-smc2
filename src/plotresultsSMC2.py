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
import os, os.path
from src.plotresults import PlotResults

class PlotResultsSMC2(PlotResults):
    def __init__(self, resultsfolder, RDatafile):
        self.method = "SMC2"
        self.color = "blue"
        PlotResults.__init__(self, resultsfolder, RDatafile)
        self.Rcode += """pdf(file = pdffile, useDingbats = FALSE, title = "%s results")\n""" % self.method
        self.parametersHaveBeenLoaded = False

    def everything(self):
        self.acceptancerate()
        self.ESS()
        self.addComputingTime()
        self.addEvidence()
        self.allParameters()
        self.addObservations()
        self.addPredictedObs()
        self.addFiltered()
        self.close()

    def singleparameter(self, parameterindex):
        self.histogramparameter(parameterindex)

    def loadparameters(self):
        self.Rcode += \
"""
indexhistory <- length(savingtimes)
t <- savingtimes[indexhistory]
w <- weighthistory[indexhistory,]
w <- w / sum(w)
#thetas <- as.data.frame(t(thetahistory[indexhistory,,]))
#thetasDF <- cbind(thetas, w)
#names(thetasDF) <- c(paste("Theta", 1:(nbparameters), sep = ""), "w")
"""
        self.parametersHaveBeenLoaded = True
    def acceptancerate(self):
        self.Rcode += \
"""
acceptratiodataframe <- as.data.frame(cbind(resamplingindices, acceptratios))
g <- ggplot(data = acceptratiodataframe, aes(x = resamplingindices, y= acceptratios))
g <- g + geom_point(size = 4) + geom_line() + xlab("iterations") + ylab("acceptance rates")
g <- g + xlim(0, T) + ylim(0, 1) 
print(g)
"""
    def ESS(self):
        self.Rcode += \
"""
Ntheta <- dim(thetahistory)[3]
ESSdataframe <- as.data.frame(cbind(1:length(ESS), ESS))
g <- ggplot(data = ESSdataframe, aes(x = V1, y= ESS))
g <- g + geom_line() + xlab("iterations") + ylab("ESS") + ylim(0, Ntheta)
print(g)
g <- ggplot(data = ESSdataframe, aes(x = V1, y= ESS))
g <- g + geom_line() + xlab("iterations (square root scale)") + ylab("ESS (log)") + ylim(0, Ntheta)
g <- g + scale_x_sqrt() + scale_y_log()
print(g)
"""
    def addEvidence(self):
        if not("No evidence" in self.plottingInstructions):
            self.Rcode += \
"""
evidencedataframe <- as.data.frame(cbind(1:length(evidences), evidences))
g <- ggplot(data = evidencedataframe, aes(x = V1, y= evidences))
g <- g + geom_line() + xlab("iterations") + ylab("evidence")
print(g)
"""

    def histogramparameter(self, parameterindex):
        if not(self.parametersHaveBeenLoaded):
            self.loadparameters()
        self.Rcode += \
"""
indexhistory <- length(savingtimes)
w <- weighthistory[indexhistory,]
w <- w / sum(w)
i <- %(parameterindex)i
g <- qplot(x = thetahistory[indexhistory,i,], weight = w, geom = "blank") + 
  geom_histogram(aes(y = ..density..)) + geom_density(fill = "blue", alpha = 0.5) +
    xlab(%(parametername)s)
""" % {"parameterindex": parameterindex + 1, "parametername": self.parameternames[parameterindex], "color": self.color}
        if hasattr(self.modeltheta, "truevalues"):
            self.Rcode += \
"""
g <- g + geom_vline(xintercept = trueparameters[i], linetype = 2, size = 1)
"""
        if hasattr(self.modeltheta, "Rprior"):
            self.Rcode += \
"""
%s
g <- g + stat_function(fun = priorfunction, aes(colour = "prior"), linetype = 1, size = 1)
if (exists("marginals")){
    g <- g + stat_function(fun = marginals[[i]], n = 50, aes(colour = "posterior"), linetype = 1, size = 1)
}
g <- g + scale_colour_discrete(name = "")
""" % self.modeltheta.Rprior[parameterindex]
#            if hasattr(self.modelx, "Rlikelihood"):
#                self.Rcode += \
#"""
#%s
#trueposterior <- function(x) priorfunction(x) * truelikelihood(x)
#g <- g + stat_function(fun = trueposterior, colour = "green", size = 2)
#""" % self.modelx.Rlikelihood[parameterindex]
        self.Rcode += \
"""
print(g)
"""
    def addComputingTime(self):
        self.Rcode += \
"""
g <- qplot(x = 1:T, y = cumsum(computingtimes), geom = "line",
           ylab = "computing time (square root scale)", xlab = "iteration")
g <- g + scale_y_sqrt()
print(g)
"""
    def addFiltered(self):
        self.Rcode += \
"""
if (exists("truestates")){
    if (is.null(dim(truestates))){
        truestates <- as.matrix(truestates, ncol = 1)
    }
}
filteredquantities <- grep(patter="filtered", x = ls(), value = TRUE)
if (length(filteredquantities) > 0){
    if (exists("filteredfirststate")){
        if (exists("kalmanresults")){
            g <- qplot(x = 1:T, y = filteredfirststate, geom = "line", colour = "SMC2 mean")
            g <- g + geom_line(aes(y = kalmanresults$FiltStateMean, colour = "KF mean"), alpha = 1.)
            g <- g + geom_point(aes(y = kalmanresults$FiltStateMean, colour = "KF mean"))
        } else {
            g <- qplot(x = 1:T, y = filteredfirststate, geom = "line", colour = "SMC2 mean")
            if (exists("truestates")){
                g <- g + geom_line(aes(y = truestates[,1], colour = "True states"))
            }
        }
        g <- g + xlab("time") + ylab("hidden states 1")
        g <- g + scale_colour_discrete(name = "")
        print(g)
        filteredquantities <- filteredquantities[filteredquantities != "filteredfirststate"]
        if (exists("filteredsecondstate")){
            g <- qplot(x = 1:T, y = filteredsecondstate, geom = "line", colour = "SMC2 mean")
            if (exists("truestates")){
                g <- g + geom_line(aes(y = truestates[,2], colour = "True states"))
            }
            g <- g + xlab("time") + ylab("hidden states 2")
            g <- g + scale_colour_discrete(name = "")
            print(g)
            filteredquantities <- filteredquantities[filteredquantities != "filteredsecondstate"]
        }
    }
    for (name in filteredquantities){
        g <- qplot(x = 1:T, geom = "blank") + geom_line(aes_string(y = name))
        g <- g + xlab("time") + ylab(name)
        print(g)
    }
}
"""













