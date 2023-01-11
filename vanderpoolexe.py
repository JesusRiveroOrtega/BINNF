# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 15:07:20 2023

@author: jrive
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:23:41 2022

@author: jrive
"""

import numpy as np
from NeuronV2 import Neuron
from matplotlib import pyplot as plt

class IntegrationEngine:
    def __init__(self, system_object, initial_time, final_time, step_size):
        self.update_function = system_object.UpdateFunction
        self.initial_time = initial_time
        self.final_time = final_time
        self.step_size = step_size
        self.time = 0
        self.state_variables = system_object.r
        self.UpdateStates = system_object.UpdateStates
        self.state_variables_record = []
        
    def CalculateK(self, k_step, coefficient_value):
        Kval = self.update_function(self.time + k_step, self.state_variables + coefficient_value)
        
        return Kval
        
    def CalculateStep(self):   
        
        
        K1 = self.CalculateK(0, 0)
        K2 = self.CalculateK(self.step_size / 2, self.step_size * K1/2)
        K3 = self.CalculateK(self.step_size / 2, self.step_size * K2/2)
        K4 = self.CalculateK(self.step_size, self.step_size * K3)
        
        K = self.step_size * (K1 + 2 * K2 + 2 * K3 + K4) / 6        
        
        self.state_variables = self.state_variables + K        
        self.UpdateStates(self.state_variables)      
        
        self.time += self.step_size
        
    def Integrate(self):
        
        
        while self.time < self.final_time:
            self.state_variables_record.append(self.state_variables)            
            self.CalculateStep()
            
class VanderPool:
    def __init__(self):
        self.omega = 1.0
        self.mu = 3.0
        self.r = np.array([1, 0], float)
    
    def UpdateFunction(self, t, state):
        """ vectorized function for the van der Pol oscillator """
        
        x = state[0]
        v = state[1]
        fx = v
        fv = -self.omega**2 * x  +  self.mu*(1 - x**2)*v
        return np.array([fx,fv], float)
    
    def UpdateStates(self, new_states):
        
        self.r = new_states

vanderpool = VanderPool()
integrator_object = IntegrationEngine(vanderpool, 0, 20, 0.02)
integrator_object.Integrate()

plt.plot(np.array(integrator_object.state_variables_record)[:, 0])




        