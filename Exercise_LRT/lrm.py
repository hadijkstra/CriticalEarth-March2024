#!/usr/bin/env python3

# author: Koen Helwegen, k.g.helwegen@uu.nl (available for questions)

import numpy as np


class LinearResponseModel(object):
    """Linear Response Model.

    Consists of a linear response model of the climate, coupled with a simple
    exponential-growth economy model. Emissions are determined by control
    variables on abatement and mitigation.
    """

    def __init__(self,
                 noise=True,
                 starting_year=2015,
                 emission_fn=None,
                 concentration_fn=None):
        """Check input and initialize object.

        Note that if a concentration function is provided, the economic
        module in the model will be ignored and the external function
        will be used instead.

        Similarly, if a concentration function is provided, the economy and
        the carbon module will be ignored and only the temperature module
        is executed.

        Args:
          noise: True for stochastic model, False for deterministic model
          starting_year: initial year (used to select initial conditions)
          emission_fn: function of the form f(t), where t is years since
            starting_year, that returns emissions in GtC.
          concentration_fn: function of the form f(t), where t is years since
            starting_year, that returns CO2 concentration in ppm.
        """
        # check input
        if starting_year not in [1765, 2005, 2015]:
            raise ValueError("Starting year must be 1765, 2005 or 2015.")
        if starting_year != 2015:
            if emission_fn is None and concentration_fn is None:
                raise ValueError("For starting years other than 2015, "
                                 "either an emission function or a"
                                 "concentration function must be provided.")

        # settings
        self.t_max = 600
        # initialize attributes
        self.time = None
        self.Ccum = None
        self.Cp = None
        self.C = None
        self.T = None
        self.noise = noise
        self.starting_year = starting_year
        self.emission_fn = emission_fn
        self.concentration_fn = concentration_fn
        # set initial state
        self.reset()

    def reset(self):
        """Reset to initial conditions."""
        self.time = 0
        self.Ccum = 0.0
        if self.starting_year == 1765:
            self.Cp = 278.05158
            self.C = [0.0, 0.0, 0.0]
            self.T = [0.0, 0.0, 0.0]
        if self.starting_year == 2005:
            self.Cp = 309.941  # permanent carbon reservoir
            self.C = [30.282, 21.013, 3.974]  # declining carbon reservoirs
            self.T = [0.085, 0.328, 0.496]  # temperature
        if self.starting_year == 2015:
            self.Cp = 319.281  # permanent carbon reservoir
            self.C = [39.048, 26.719, 5.216]  # declining carbon reservoirs
            self.T = [0.110, 0.408, 0.617]  # temperature

    def step(self, abatement=None, mitigation=None):
        """Step model one year forward.

        Args:
          abatement: abatement rate (in [0,1])
          mitigation: mitigation rate (in [0,1])

        Returns:
          C: current atmospheric CO2 concentration in ppm
          T: current temperature (Kelvin difference with preindustrial)
          Ccum: cummulative emissions since starting year
        """
        # parameters
        a_p = 0.2173  # part carbon that stays permanently in the atmosphere
        a = [0.2240, 0.2824, 0.2763]  # distribution carbon over reservoirs
        b = [0.00115176, 0.10967972, 0.03361102]  # distribution temperature
        tau_a = [394.4, 36.54, 4.304]  # decline rates carbon
        tau_b = [400.0, 1.42706247, 8.02118539] # decline rates temperature
        A = 1.48  # upscaling of radiatiave forcing for non-co2 GHG's
        alpha = 5.35  # alpha/ln(2) is the climate senstivity
        C0 = 278.0  # preindustrial CO2 concentration
        y0 = 73.0  # initial size of the economy
        g = 0.02  # growth rate of the economy
        gamma_0 = 1.4e-4  # initial energy efficiancy
        r_gamma = 0.0  # rate of change of energy efficiancy
        sigma_c2 = .65  # noise on C[1]
        sigma_t0 = 0.015  # noise on T[0]
        sigma_t2 = 0.13  # noise on T[2]

        # if neither emissions nor concentrations are provided,
        # run economy and carbon module:
        if self.emission_fn is None and self.concentration_fn is None:
            # economy
            y = y0*np.exp(g*self.time)
            # We scale the energy to make sure cummulative emissions are in
            # proper range (this doesn't affect the climate model):
            en = 0.55*1e3*gamma_0*np.exp(-r_gamma*self.time)*y
            e = (1-abatement)*(1-mitigation)*en
            self.Ccum += e
            # current atmospheric carbon concentration
            C = (self.Cp + sum(self.C))
            # carbon model
            self.Cp += a_p * e
            self.C = [
                a[ii] * e * tau_a[ii] + (self.C[ii] - a[ii] * e * tau_a[ii])
                * np.exp(-1 / tau_a[ii]) for ii in range(3)]

        # if emissions are provided (but not concentrations), run
        # carbon module:
        elif self.concentration_fn is None:
            e = self.emission_fn(self.time)
            self.Ccum += e
            # current atmospheric carbon concentration
            C = (self.Cp + sum(self.C))
            # carbon model
            self.Cp += a_p*e
            self.C = [a[ii]*e*tau_a[ii] + (self.C[ii]-a[ii]*e*tau_a[ii])
                      * np.exp(-1/tau_a[ii]) for ii in range(3)]

        # otherwise, get concentrations:
        else:
            C = self.concentration_fn(self.time)

        # Run temperature module:

        # radiative forcing
        f = A*alpha*np.log(C/C0)
        # temperature model
        self.T = [b[ii]*f*tau_b[ii] + (self.T[ii]-b[ii]*f*tau_b[ii])
                  * np.exp(-1/tau_b[ii]) for ii in range(3)]
        # add noise
        if self.noise:
            self.C[1] += sigma_c2*np.random.randn()
            self.T[0] += sigma_t0*np.random.randn()
            self.T[2] += sigma_t2*np.random.randn()*self.T[2]
        # update time
        self.time += 1

        # current state
        C = (self.Cp + sum(self.C))
        T = sum(self.T)

        return C, T, self.Ccum

    def getstate(self):
        # current state
        C = (self.Cp + sum(self.C))
        T = sum(self.T)

        return C, T, self.Ccum
