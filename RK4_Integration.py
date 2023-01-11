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
from Neuron import Neuron
from matplotlib import pyplot as plt

class IntegrationEngine:
    def __init__(self, system_object, initial_time, final_time, step_size):
        '''
        

        Parameters
        ----------
        system_object : Integrand object
            Object created with the class that represents the system to be integrated.
        initial_time : float
            Starting time point.
        final_time : float
            Final time point.
        step_size : float
            Change of time in each update step.

        Returns
        -------
        None.

        '''
        self.update_function = system_object.UpdateFunction
        self.initial_time = initial_time
        self.final_time = final_time
        self.step_size = step_size
        self.time = 0
        self.state_variables = system_object.current_firing_rate
        self.UpdateStates = system_object.UpdateStates
        self.state_variables_record = []
        
        
    def CalculateK(self, k_step, coefficient_value):
        '''
        

        Parameters
        ----------
        k_step : float
            DESCRIPTION.
        coefficient_value : TYPE
            DESCRIPTION.

        Returns
        -------
        Kval : TYPE
            DESCRIPTION.

        '''
        
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
            
    
            


tanks_semisaturation_values = 50 * np.ones((2, 1))
tanks_max_saturation_values = 100 * np.ones((2, 1))
tanks_tau = 0.2 * np.ones((2, 1))

rings_semisaturation_values = 50 * np.ones((48,  1))
rings_max_saturation_values = 100 * np.ones((48, 1))
rings_tau = 1.5 * np.ones((48, 1))


sm_values = np.vstack((rings_semisaturation_values, tanks_semisaturation_values))
mm_values = np.vstack((rings_max_saturation_values, tanks_max_saturation_values))
tau_values = np.vstack((rings_tau, tanks_tau))

print("Integrator Engine")            
neuron_object = Neuron(50, sm_values, mm_values, 2, tau_values) 


integrator_object = IntegrationEngine(neuron_object, 0, 40, 0.01)
integrator_object.Integrate()


neuron_object.PlotRingActivity(range(0, 12), integrator_object.state_variables_record, 1, "Current Orientation Ring", 12)
neuron_object.PlotRingActivity(range(12, 24), integrator_object.state_variables_record, 2, "Setpoint Orientation Ring", 12)   
neuron_object.PlotRingActivity(range(24, 36), integrator_object.state_variables_record, 3, "Current minus Setpoint Activity", 12)    
neuron_object.PlotRingActivity(range(36, 48), integrator_object.state_variables_record, 4, "Setpoint minus Current Activity", 12) 
neuron_object.PlotRingActivity(48, integrator_object.state_variables_record, 5, "Rotate to the Right Tank Activity", 12)    
neuron_object.PlotRingActivity(49, integrator_object.state_variables_record, 6, "Rotate to the Left Tank Activity", 12)  

plt.plot(neuron_object.DecodeOrientation(integrator_object.state_variables_record, range(0, 12), 12))





        