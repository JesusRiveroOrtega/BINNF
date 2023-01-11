# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:23:41 2022

@author: jrive
"""

import numpy as np
from NeuronV2 import Neuron
from matplotlib import pyplot as plt

class IntegrationEngine:
    def __init__(self, neuron_object, initial_time, final_time, step_size):
        self.update_function = neuron_object.CalculateFiringRate
        self.initial_time = initial_time
        self.final_time = final_time
        self.step_size = step_size
        self.firing_rate_values = []
        self.time_values = []
        self.stim_record = []
        self.stim = neuron_object.presynaptic_potential
        self.current_firing_rate = neuron_object.current_firing_rate
        
    def CalculateK(self, coefficient_value):
        Kval = self.step_size * self.update_function(self.current_firing_rate + coefficient_value)
        
        return Kval
        
    def CalculateStep(self, time):   
        
        
        K1 = self.CalculateK(0)
        K2 = self.CalculateK(self.step_size * K1/2)
        K3 = self.CalculateK(self.step_size * K2/2)
        K4 = self.CalculateK(self.step_size * K3)
        
        K = (K1 + 2 * K2 + 2 * K3 + K4) / 6
        
        
        self.current_firing_rate = self.current_firing_rate + K
        #print("FR at: ", time, "\n", self.current_firing_rate[0:12, :].T)
        neuron_object.current_firing_rate = self.current_firing_rate + K
        
        
        
        
        return time + self.step_size
        
    def Integrate(self):
        
        t = 0
        while t < self.final_time:    
            self.firing_rate_values.append(self.current_firing_rate)
            
            t = self.CalculateStep(t)            
            
            self.time_values.append(t)            
            self.stim_record.append(self.stim)
            
            #print("FR at t: ", t, " - ", self.current_firing_rate[range(0, 12), :])
            #neuron_object.UpdateCurrentOrientation(0.1, 0.1)
            

def PlotRingActivity(index_ring, neuron_obj, fig_num, title):
    ring_activity = np.array(integrator_object.firing_rate_values)[:,index_ring]
    #t = np.array(integrator_object.time_values)
    plt.figure(num = fig_num)
    plt.title(title)
    plt.plot(ring_activity.squeeze())
    plt.legend(np.round(np.degrees(np.linspace(-np.pi, np.pi, 12, endpoint = False))))
    plt.show()
    
def PlotNeuronsActivity(index_ring, neuron_obj, fig_num, title):
    ring_activity = np.array(integrator_object.firing_rate_values)[:,index_ring]
    #t = np.array(integrator_object.time_values)
    plt.figure(num = fig_num)
    plt.title(title)
    plt.plot(ring_activity.squeeze())
    
    plt.show()
    
def DecodeOrientation(index_ring, integrator_object):
    a = (np.array(integrator_object.firing_rate_values)).squeeze()[:,index_ring]
    b = np.linspace(-np.pi, np.pi, 12, endpoint = False)
    #print("a: ", a.shape, "b: ", b.shape, "a*b: ", (a*b).shape )
    val = np.sum(a*b, axis=1)/np.sum(np.array(integrator_object.firing_rate_values)[:,index_ring])
    #print(val.shape)
    return np.degrees(val)
        



tanks_semisaturation_values = 50 * np.ones((2, 1))
tanks_max_saturation_values = 100 * np.ones((2, 1))
tanks_tau = 1 * np.ones((2, 1))

rings_semisaturation_values = 50 * np.ones((48,  1))
rings_max_saturation_values = 100 * np.ones((48, 1))
#rings_tau = 1.5 * np.ones((48, 1))
rings_tau = 0.2 * np.ones((48, 1))

sm_values = np.vstack((rings_semisaturation_values, tanks_semisaturation_values))
mm_values = np.vstack((rings_max_saturation_values, tanks_max_saturation_values))
tau_values = np.vstack((rings_tau, tanks_tau))

print("Integrator Engine")            
neuron_object = Neuron(50, sm_values, mm_values, 2, tau_values) 
#neuron_object = Neuron(2, 50, 100, 2, 1.5) 

integrator_object = IntegrationEngine(neuron_object, 0, 4, 0.01)
integrator_object.Integrate()



PlotRingActivity(range(0, 12), neuron_object, 1, "Current Orientation Ring")    
PlotRingActivity(range(12, 24), neuron_object, 2, "Setpoint Orientation Ring")   
PlotRingActivity(range(24, 36), neuron_object, 3, "Current minus Setpoint Activity")    
PlotRingActivity(range(36, 48), neuron_object, 4, "Setpoint minus Current Activity") 
PlotRingActivity(48, neuron_object, 5, "Rotate to the Left Tank Activity")    
PlotRingActivity(49, neuron_object, 6, "Rotate to the Right Tank Activity")     

#plt.plot(neuron_object.current_orientation_list)
#plt.title("Current orientation")

#PlotNeuronsActivity(range(0, 2), neuron_object, 1, "Current Orientation Ring")    




        
        
        