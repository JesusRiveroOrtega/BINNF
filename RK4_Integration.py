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
            
    
            
units_per_ring = 24

tanks_semisaturation_values = 50 * np.ones((2, 1))
tanks_max_saturation_values = 100 * np.ones((2, 1))
tanks_tau = 0.1 * np.ones((2, 1))

rings_semisaturation_values = 50 * np.ones((units_per_ring * 6,  1))
rings_max_saturation_values = 100 * np.ones((units_per_ring * 6, 1))
rings_tau = 0.1 * np.ones((units_per_ring * 6, 1))


sm_values = np.vstack((rings_semisaturation_values, tanks_semisaturation_values))
mm_values = np.vstack((rings_max_saturation_values, tanks_max_saturation_values))
tau_values = np.vstack((rings_tau, tanks_tau))

print("Integrator Engine")            
neuron_object = Neuron(units_per_ring * 6 + 2, sm_values, mm_values, 2, tau_values, units_per_ring) 


integrator_object = IntegrationEngine(neuron_object, 0, 70, 0.01)
integrator_object.Integrate()




# r_number_input = 0
# ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
# neuron_object.PlotRingActivity(ring_indexes_input, integrator_object.state_variables_record, 1, "Target Ring")
# r_number_input = 1
# ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
# neuron_object.PlotRingActivity(ring_indexes_input, integrator_object.state_variables_record, 2, "Obstacles Ring")   
# r_number_input = 2
# ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
# neuron_object.PlotRingActivity(ring_indexes_input, integrator_object.state_variables_record, 3, "Setpoint Ring")    
# r_number_input = 3
# ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
# neuron_object.PlotRingActivity(ring_indexes_input, integrator_object.state_variables_record, 4, "Current Ring") 
# r_number_input = 4
# ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
# neuron_object.PlotRingActivity(ring_indexes_input, integrator_object.state_variables_record, 5, "Current minus Setpoint Activity")
# r_number_input = 5
# ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
# neuron_object.PlotRingActivity(ring_indexes_input, integrator_object.state_variables_record, 6, "Setpoint minus Current Activity") 

r_number = 5
tank_index = (r_number + 1) * units_per_ring
neuron_object.PlotRingActivity(tank_index, integrator_object.state_variables_record, 7, "Rotate to the Right Tank Activity")

r_number = 5
tank_index = (r_number + 1) * units_per_ring + 1
neuron_object.PlotRingActivity(tank_index, integrator_object.state_variables_record, 8, "Rotate to the Left Tank Activity")  


r_number_input = 0
ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
neuron_object.plot_circles(ring_indexes_input)
plt.title("Targets Layer")


r_number_input = 1
ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input +  1) * units_per_ring)
neuron_object.plot_circles(ring_indexes_input)
plt.title("Obstacles Layer")


# r_number_input = 2
# ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
# neuron_object.plot_circles(ring_indexes_input)
# plt.title("Setpoint Layer")


# r_number_input = 3
# ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
# neuron_object.plot_circles(ring_indexes_input)
# plt.title("Current Ring")
# plt.show()


r_number_setpoint = 2
ring_indexes_setpoint = range(r_number_setpoint * units_per_ring, (r_number_setpoint + 1) * units_per_ring)

r_number_current = 3
ring_indexes_current = range(r_number_current * units_per_ring, (r_number_current + 1) * units_per_ring)

neuron_object.current_setpoint_circles(ring_indexes_current, ring_indexes_setpoint)
plt.show()