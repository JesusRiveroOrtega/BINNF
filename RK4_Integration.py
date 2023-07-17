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
        self.update_function = system_object.UpdateFunction
        self.state_variables = system_object.current_firing_rate
        self.UpdateStates = system_object.UpdateStates
        self.system_object = system_object
        self.update_current_orientation = system_object.UpdateCurrentOrientation
        self.initial_time = initial_time
        self.final_time = final_time
        self.step_size = step_size        
        self.time_values = []
        self.stim_record = []
        self.state_variables_record = []
        self.time = 0
        self.misc_record = []
        
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
        
        self.system_object.SetCurrentOrientation(0)
        
        while self.time < self.final_time:

            if self.time <= 0.5:            
                self.system_object.SetDistance_to_Target(50)
                self.system_object.SetTargetOrientation(90)
                #self.system_object.SetObstaclesOrientation([[90], [2]])
            elif self.time > 0.5 and self.time <= 1.0:
                self.system_object.SetDistance_to_Target(None)
                self.system_object.SetTargetOrientation(None)
                self.system_object.SetCurrentOrientation(None)
                #self.system_object.SetObstaclesOrientation(None)
            # elif self.time > 1.0 and self.time <= 1.5:
            #     r_number_input = 3
            #     ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
            #     decoded_orientation = self.system_object.DecodeOrientation(self.state_variables, ring_indexes_input, units_per_ring)
            #     self.system_object.SetCurrentOrientation(decoded_orientation)                
            # elif self.time > 1.5 and self.time <= 2.0:
            #     self.system_object.UpdateCurrentOrientation(0.0005, 0.0005)            
            # else: 
            #     self.system_object.SetCurrentOrientation(None)

            r_number_input = 3
            ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)    
            self.misc_record.append(self.system_object.DecodeOrientation(self.state_variables, ring_indexes_input, units_per_ring))

            self.state_variables_record.append(self.state_variables)            
            self.CalculateStep()                        
            self.time_values.append(self.time)            
              
            
    
            
units_per_ring = 24  
tanks_semisaturation_values = 40 * np.ones((5, 1))
tanks_max_saturation_values = 100 * np.ones((5, 1))
tanks_tau = 0.001 * np.ones((5, 1))
rings_semisaturation_values = 40 * np.ones((units_per_ring * 6,  1))
rings_max_saturation_values = 100 * np.ones((units_per_ring * 6, 1))
rings_tau = 0.001 * np.ones((units_per_ring * 6, 1))
sm_values = np.vstack((rings_semisaturation_values, tanks_semisaturation_values))
mm_values = np.vstack((rings_max_saturation_values, tanks_max_saturation_values))
tau_values = np.vstack((rings_tau, tanks_tau))

print("Integrator Engine")            
neuron_object = Neuron(units_per_ring * 6 + 5, sm_values, mm_values, 2, tau_values, units_per_ring, np.array((1,20)))

integrator_object = IntegrationEngine(neuron_object, 0, 1, 0.0001)
integrator_object.Integrate()


# r_number_input = 0
# ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
# neuron_object.PlotRingActivity(ring_indexes_input, integrator_object.state_variables_record, integrator_object.time_values, 1, "Target Encoding Layer")
# r_number_input = 1
# ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
# neuron_object.PlotRingActivity(ring_indexes_input, integrator_object.state_variables_record, integrator_object.time_values, 2, "Obstacles Encoding Layer")   
r_number_input = 2
ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
neuron_object.PlotRingActivity(ring_indexes_input, integrator_object.state_variables_record, integrator_object.time_values, 3, "Setpoint Orientation Ring")    
r_number_input = 3
ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
neuron_object.PlotRingActivity(ring_indexes_input, integrator_object.state_variables_record, integrator_object.time_values, 4, "Current Orientation Ring") 
# r_number_input = 4
# ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
# neuron_object.PlotRingActivity(ring_indexes_input, integrator_object.state_variables_record, 5, "Current minus Setpoint Activity")
# r_number_input = 5
# ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
# neuron_object.PlotRingActivity(ring_indexes_input, integrator_object.state_variables_record, 6, "Setpoint minus Current Activity") 

# r_number = 5
# tank_index = (r_number + 1) * units_per_ring
# neuron_object.PlotRingActivity(tank_index, integrator_object.state_variables_record, 7, "Rotate to the Right Tank Activity")

# r_number = 5
# tank_index = (r_number + 1) * units_per_ring + 1
# neuron_object.PlotRingActivity(tank_index, integrator_object.state_variables_record, 8, "Rotate to the Left Tank Activity")  


r_number_input = 0
ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
target_color = (225/255, 128/255, 0/255)
neuron_object.plot_circles(ring_indexes_input, target_color)
plt.title("Targets Layer")


r_number_input = 1
ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input +  1) * units_per_ring)
obstacles_color = (225/255, 0/255, 0/255)
neuron_object.plot_circles(ring_indexes_input, obstacles_color)
plt.title("Obstacles Layer")


r_number_input = 2
ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
setpoint_color = (225/255, 0/255, 152/255)
neuron_object.plot_circles(ring_indexes_input, setpoint_color)
plt.title("Setpoint Orientation Layer")


r_number_input = 3
ring_indexes_input = range(r_number_input * units_per_ring, (r_number_input + 1) * units_per_ring)
current_color = (127/255,255/255,212/255)
neuron_object.plot_circles(ring_indexes_input, current_color)
plt.title("Current Orientation Ring")

# r_number_setpoint = 2
# ring_indexes_setpoint = range(r_number_setpoint * units_per_ring, (r_number_setpoint + 1) * units_per_ring)
# setpoint_color = (225/255, 0/255, 152/255)
# r_number_current = 3
# ring_indexes_current = range(r_number_current * units_per_ring, (r_number_current + 1) * units_per_ring)
# current_color = (127/255,255/255,212/255)
# neuron_object.current_setpoint_circles(ring_indexes_current, ring_indexes_setpoint, current_color, setpoint_color)
# plt.title("Setpoint Orientation Ring and Current Orientation Ring")

plt.figure()
plt.plot(integrator_object.time_values, integrator_object.misc_record, linewidth = 3)
plt.ylim([-180, 180])
angles = np.linspace(-180, 180, 24, endpoint=False)
for i in range(len(angles)):
    plt.axhline(y = angles[i], xmin = 0, xmax = integrator_object.time_values[-1], linewidth = 1, color = "black", linestyle = ":")
plt.yticks(angles)
plt.title("Decoded Current Orientation")

plt.show()