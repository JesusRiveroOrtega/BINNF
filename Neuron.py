# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 19:08:46 2022

@author: jrive
"""
import numpy as np
from matplotlib import pyplot as plt

class Neuron:
    def __init__(self, n_neurons, semisaturation, max_saturation, input_exp, tau, units_per_ring):
        
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
        self.current_orientation = 180
        self.setpoint_orientation = 90
        self.units_per_ring = units_per_ring
        
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
    
    def SetGaussianStim(self, ring_index, input_angle, area, min_value, max_value, n_units):
        input_angle = np.deg2rad(input_angle)
        if input_angle < -np.pi or input_angle > np.pi:
            input_angle -= (np.sign(input_angle) * np.pi * np.ceil(np.abs(input_angle / np.pi)))

        angles = np.linspace(-2*np.pi, 2*np.pi, 2*n_units, endpoint=False)
        gaussian_projection_pre = min_value + (max_value - min_value) * np.exp((-1/2) * ((angles - input_angle)**2 / (area)**2))
        
        gaussian_projection = gaussian_projection_pre[int(n_units/2) : int(n_units/2) + n_units]
        gaussian_projection[int(n_units/2) : int(n_units/2) + n_units] += gaussian_projection_pre[ : int(n_units/2)]
        gaussian_projection[ : int(n_units/2)] += gaussian_projection_pre[int(n_units/2) + n_units : ]
        
        self.presynaptic_potential[ring_index] += np.expand_dims(gaussian_projection, axis=1)

    def SetConnections(self): 
        
        # Target Layer ---------------------------------------------------------------------------------
        ## Ring Number = 0
        r_number = 0
        
        ring_indexes = range(r_number * self.units_per_ring, (r_number + 1) * self.units_per_ring)
        self.SetGaussianStim(ring_indexes, self.setpoint_orientation, 0.4, 50, 100, self.units_per_ring)

        # Obstacles Layer
        ## Ring Number = 1
        r_number = 1
        
        ring_indexes = range(r_number * self.units_per_ring, (r_number + 1) * self.units_per_ring)
        self.SetGaussianStim(ring_indexes, 60, 0.3, 0, 100, self.units_per_ring)
        self.SetGaussianStim(ring_indexes, 120, 0.3, 0, 100, self.units_per_ring)


        

        # Setpoint Layer ---------------------------------------------------------------------------------
        # Define this ring's indexes
        r_number = 2
        
        ring_indexes = range(r_number * self.units_per_ring, (r_number + 1) * self.units_per_ring)


        # Define inputs to the layer
        # Input 1 (Target Layer)
        r_number_input = 0
        
        ring_indexes_input = range(r_number_input * self.units_per_ring, (r_number_input + 1) * self.units_per_ring)
        # Add first input
        self.DirectConnection(ring_indexes, ring_indexes_input, 1.0)

        # Input 2 (Obstacles Layer)
        r_number_input = 1
        
        ring_indexes_input = range(r_number_input * self.units_per_ring, (r_number_input + 1) * self.units_per_ring)
        # Add second input
        self.DirectConnection(ring_indexes, ring_indexes_input, -2.0)
        
        # Define self connection of the layer
        setpoint_pattern = -100*np.array([1, 1, 1, 1, 1, -0.01, 1, 1, 1, 1, 1])
        self.RingSelfConnections(setpoint_pattern, ring_indexes)

        # Current Orientation Layer ---------------------------------------------------------------------------------
        # Inputs for current orientation ring
        r_number = 3

        ring_indexes = range(r_number * self.units_per_ring, (r_number + 1) * self.units_per_ring)


        preferred_direction = np.linspace(-np.pi, np.pi, self.units_per_ring, endpoint = False)
        self.PreferentialOrientation(ring_indexes, preferred_direction, self.current_orientation)        
        current_pattern = -1*np.array([1, 1, 1, 1, -0.5, -1, -0.5, 1, 1, 1, 1])
        #self.RingSelfConnections(current_pattern, ring_indexes)
        

        # Current minus Setpoint Layer ---------------------------------------------------------------------------------
        r_number = 4
        
        ring_indexes = range(r_number * self.units_per_ring, (r_number + 1) * self.units_per_ring)

        # Define input from Current Orientation Layer
        r_number_input = 3
        
        ring_indexes_input = range(r_number_input * self.units_per_ring, (r_number_input + 1) * self.units_per_ring)

        self.DirectConnection(ring_indexes, ring_indexes_input, np.ones((self.units_per_ring, 1)))

        # Define input from Setpoint Layer
        r_number_input = 2
        
        ring_indexes_input = range(r_number_input * self.units_per_ring, (r_number_input + 1) * self.units_per_ring)

        cur_set_pattern = -1*np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.RingOutConnections(cur_set_pattern, ring_indexes, ring_indexes_input)


        # Setpoint minus Current Layer ---------------------------------------------------------------------------------
        r_number = 5
        
        ring_indexes = range(r_number * self.units_per_ring, (r_number + 1) * self.units_per_ring)

        # Define input from Setpoint Layer
        r_number_input = 2
        
        ring_indexes_input = range(r_number_input * self.units_per_ring, (r_number_input + 1) * self.units_per_ring)
        
        self.DirectConnection(ring_indexes, ring_indexes_input, np.ones((self.units_per_ring, 1)))

        # Define input from Current Orientation Layer
        r_number_input = 3
        
        ring_indexes_input = range(r_number_input * self.units_per_ring, (r_number_input + 1) * self.units_per_ring)

        set_cur_pattern = -1*np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.RingOutConnections(set_cur_pattern, ring_indexes, ring_indexes_input)

        # Tank 1 ---------------------------------------------------------------------------------
        # Define input from Current minus Setpoint Layer
        r_number_input = 4
        
        ring_indexes_input = range(r_number_input * self.units_per_ring, (r_number_input + 1) * self.units_per_ring)
        self.TankConnections(np.ones((self.units_per_ring, 1)), 72, ring_indexes_input)

        # Tank 2 ---------------------------------------------------------------------------------
        # Define input from Setpoint minus Current Layer
        r_number_input = 5
        
        ring_indexes_input = range(r_number_input * self.units_per_ring, (r_number_input + 1) * self.units_per_ring)
        self.TankConnections(np.ones((self.units_per_ring, 1)), 73, ring_indexes_input)


        
        
    
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
        #self.UpdateCurrentOrientation(10, 10)
    
        
        
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
        
        self.current_orientation += (-alpha*self.current_firing_rate[48,:][0] + beta*self.current_firing_rate[49,:][0])
        if self.current_orientation < 0:
            self.current_orientation += 360
            
        if self.current_orientation > 360:
            self.current_orientation -= 360
            
        self.current_orientation_list.append(self.current_orientation)
        #print("Output sum", alpha*self.current_firing_rate[48,:] - beta*self.current_firing_rate[49,:][0])
        print("Current Orientation", self.current_orientation)
        
        
    def PlotRingActivity(self, index_ring, ring_firing_rates, fig_num, title, n_neurons_ring):
        ring_activity = np.array(ring_firing_rates)[:,index_ring]
        #t = np.array(integrator_object.time_values)
        plt.figure(num = fig_num)
        plt.title(title)
        plt.plot(ring_activity.squeeze())
        plt.legend(np.round(np.degrees(np.linspace(-np.pi, np.pi, n_neurons_ring, endpoint = False))))
        
        
    def PlotNeuronsActivity(self, index_neuron, neuron_firing_rates, fig_num, title):
        neuron_activity = np.array(neuron_firing_rates)[:,index_neuron]
        #t = np.array(integrator_object.time_values)
        plt.figure(num = fig_num)
        plt.title(title)
        plt.plot(neuron_activity.squeeze())
        
        
        
    def DecodeOrientation(self, activity, ring_index, n_neurons_ring):
        activity = np.array(activity).squeeze()[:, ring_index]
        orientations = np.linspace(-np.pi, np.pi, n_neurons_ring, endpoint = False)
        return np.rad2deg(np.sum(activity * orientations, axis = 1) / np.sum(activity, axis = 1))
        
    def plot_circles(self, index_ring):     

        # Create a figure and axes
        fig, ax = plt.subplots()

        # Set the axis limits to be square
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)

        # Create an array of angles for the circles to be plotted at
        angles = np.linspace(-180, 180, len(index_ring), endpoint = False)

        # Plot the circles
        for angle, sat in zip(angles, (((self.current_firing_rate[index_ring]) / np.amax(self.current_firing_rate[index_ring]))).tolist()):
            x = 20 * np.cos(np.deg2rad(angle))
            y = 20 * np.sin(np.deg2rad(angle))
            
            circle = plt.Circle((x, y), radius=2.65, facecolor = (0.3, 0.4, 0.6, sat[0]), edgecolor="white", linewidth = 2)
            text = plt.text(x, y, str(int(angle)), fontsize = 9, color = (1, 1, 1))
            
            ax.add_artist(circle)
            ax.add_artist(text)
        fig.set_size_inches((8, 8))
        ax.set_facecolor((0,0,0))

        

        
   
        

                
        