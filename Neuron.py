# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 19:08:46 2022

@author: jrive
"""
import numpy as np
from matplotlib import pyplot as plt

class Neuron:
    def __init__(self, n_neurons, semisaturation, max_saturation, input_exp, tau):
        
        self.semisaturation = semisaturation * np.ones((n_neurons, 1))
        self.max_saturation = max_saturation * np.ones((n_neurons, 1))
        self.input_exp = input_exp
        self.adaptation = np.zeros((n_neurons, 1))
        self.tau = tau * np.ones((n_neurons, 1))
        self.firing_rate = np.zeros((n_neurons, 1))
        self.presynaptic_potential = np.zeros((n_neurons, 1))
        self.current_firing_rate = np.zeros((n_neurons, 1))
        self.n_neurons = n_neurons
        self.current_orientation_list = []
        self.current_orientation = 45
        self.setpoint_orientation = 90
        
    def NakaRushton(self, presynaptic_potential):
        #print(presynaptic_potential)
        presynaptic_potential = presynaptic_potential * (presynaptic_potential > 0)        
        postsynaptic_potential = (self.max_saturation) * (presynaptic_potential ** self.input_exp) / ((self.semisaturation + self.adaptation)**self.input_exp + (presynaptic_potential ** self.input_exp))                
        return postsynaptic_potential
    
    def CalculatePreSynapticPotential(self, inputs, weights):
        return inputs*weights
        
    def CalculatePostSynapticPotential(self, presynaptic_potential):
        return self.NakaRushton(presynaptic_potential)
    
    def CalculateActivityUpdate(self, current_firing_rate, postsynaptic_potential):        
        return (1 / self.tau) * (- current_firing_rate + postsynaptic_potential)
    
    def SetStim(self, unit_index, value):
        self.presynaptic_potential[unit_index] += value

    def SetConnections(self): 
        
        #self.SetStim(0, 100)
        #self.DirectConnection(1, 0, 1)
        
        
        
        
        preferred_direction = np.linspace(-np.pi, np.pi, 12, endpoint = False)
        
        # Inputs for current orientation ring
        self.PreferentialOrientation(range(0, 12), preferred_direction, self.current_orientation)        
        current_pattern = -1*np.array([1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1])
        self.RingSelfConnections(current_pattern, range(0, 12))
        
        # Inputs for setpoint orientation ring   
        #self.PreferentialOrientation(range(12, 24), preferred_direction, self.setpoint_orientation)    
        setpoint_pattern = -1*np.array([1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1])
        self.RingSelfConnections(setpoint_pattern, range(12, 24))
        
        # Current minus Setpoint Layer Connections
        self.DirectConnection(range(24, 36), range(0, 12), np.ones((12, 1)))
        cur_set_pattern = -1*np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.RingOutConnections(cur_set_pattern, range(24, 36), range(12, 24))
        
        # Setpoint minus Current Layer Connections
        self.DirectConnection(range(36, 48), range(12, 24), np.ones((12, 1)))
        set_cur_pattern = -1*np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.RingOutConnections(set_cur_pattern, range(36, 48), range(0, 12))

        #Tank 1 Connections
        self.TankConnections(np.ones((12, 1)), 48, range(24, 36))
        
        #Tank 2 Connections
        self.TankConnections(np.ones((12, 1)), 49, range(36, 48))
        
        #Obstacles layer
        self.SetStim([12, 13, 23], -100)
        
        #Target layer
        self.SetStim(20, 100)
        
        
    
    def DirectConnection(self, n1, n2, w):        
        '''
            n1: Index of the receiving neuron
            n2: Index of the sending neuron
        '''
        self.presynaptic_potential[n1] += w * self.current_firing_rate[n2]
        
    
    def UpdateFunction(self, time, current_firing_rate):
        
        self.SetConnections()
        postsynaptic_potential = self.CalculatePostSynapticPotential(self.presynaptic_potential)
        activity_update = self.CalculateActivityUpdate(current_firing_rate, postsynaptic_potential)
        self.presynaptic_potential = np.zeros((self.n_neurons, 1))
        
        return activity_update
    
    def UpdateStates(self, new_states):
        self.current_firing_rate = new_states
        #self.UpdateCurrentOrientation(0.001, 0.001)
    
        
        
    def RingSelfConnections(self, weights, ring_index):
        
        inputs = self.current_firing_rate[ring_index, :]
        inputs = np.hstack((inputs.T, inputs.T, inputs.T))
        
        n = len(weights)
        conv = np.convolve(np.flip(weights), inputs.squeeze())
    
        conv = conv[int(np.fix(n / 2) + len(inputs)): int(np.fix(n / 2) + 2 * len(inputs))]
        self.presynaptic_potential[ring_index, :] += conv
        
    def RingOutConnections(self, weights, ring_index, input_index):
        
        inputs = self.current_firing_rate[input_index, :]
        inputs_stacked = np.hstack((inputs.T, inputs.T, inputs.T))
        
        n = len(weights)
        conv = np.convolve(np.flip(weights), inputs_stacked.squeeze())        
        conv = np.expand_dims(conv[int(np.fix(n / 2) + len(inputs)): int(np.fix(n / 2) + 2 * len(inputs))], axis=1)
        
        self.presynaptic_potential[ring_index, :] += conv
        
    def TankConnections(self, weights, neuron_index, input_index):
        self.presynaptic_potential[neuron_index] = np.sum(weights * self.current_firing_rate[input_index, :])
        
    def PreferentialOrientation(self, ring_index, preferred_direction, orientation):
        projection = 100 * np.cos(preferred_direction - np.radians(orientation))
        projection = projection * (projection > 0)
        
        self.presynaptic_potential[ring_index, :] += np.expand_dims(projection, axis = 1)
        
    def UpdateCurrentOrientation(self, alpha, beta):
        
        self.current_orientation += (-alpha*self.current_firing_rate[48,:] + beta*self.current_firing_rate[49,:][0])
        if self.current_orientation < 0:
            self.current_orientation += 360
            
        if self.current_orientation > 360:
            self.current_orientation -= 360
            
        self.current_orientation_list.append(self.current_orientation[0])
        #print("Output sum", alpha*self.current_firing_rate[48,:] - beta*self.current_firing_rate[49,:][0])
        print("Current Orientation", self.current_orientation)
        
        
    def PlotRingActivity(self, index_ring, ring_firing_rates, fig_num, title, n_neurons_ring):
        ring_activity = np.array(ring_firing_rates)[:,index_ring]
        #t = np.array(integrator_object.time_values)
        plt.figure(num = fig_num)
        plt.title(title)
        plt.plot(ring_activity.squeeze())
        plt.legend(np.round(np.degrees(np.linspace(-np.pi, np.pi, n_neurons_ring, endpoint = False))))
        plt.show()
        
    def PlotNeuronsActivity(self, index_neuron, neuron_firing_rates, fig_num, title):
        neuron_activity = np.array(neuron_firing_rates)[:,index_neuron]
        #t = np.array(integrator_object.time_values)
        plt.figure(num = fig_num)
        plt.title(title)
        plt.plot(neuron_activity.squeeze())
        
        plt.show()
        
    def DecodeOrientation(self, activity, ring_index, n_neurons_ring):
        activity = np.array(activity).squeeze()[:, ring_index]
        orientations = np.linspace(-np.pi, np.pi, n_neurons_ring, endpoint = False)
        return np.rad2deg(np.sum(activity * orientations, axis = 1) / np.sum(activity, axis = 1))
        
        
        

        
   
        

                
        