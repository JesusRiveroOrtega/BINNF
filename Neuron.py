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
        self.current_orientation = 0
        self.setpoint_orientation = 90
        self.units_per_ring = units_per_ring
        
        
    def NakaRushton(self, presynaptic_potential):
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
        self.SetGaussianStim(ring_indexes, self.setpoint_orientation, 0.2, 20, 120, self.units_per_ring)

        # Obstacles Layer
        ## Ring Number = 1
        r_number = 1        
        ring_indexes = range(r_number * self.units_per_ring, (r_number + 1) * self.units_per_ring)        
        if len(self.obstacle_orientations) != 0:
            self.SetStimObstacles(ring_indexes,  self.obstacle_orientations[0], self.obstacle_orientations[1])

        # Setpoint Layer ---------------------------------------------------------------------------------
        # Define this ring's indexes
        r_number = 2        
        ring_indexes = range(r_number * self.units_per_ring, (r_number + 1) * self.units_per_ring)

        # Define inputs to the layer
        # Input 1 (Target Layer)
        r_number_input = 0        
        ring_indexes_input = range(r_number_input * self.units_per_ring, (r_number_input + 1) * self.units_per_ring)
        # Add first input
        self.DirectConnection(ring_indexes, ring_indexes_input, 2.0)

        # Input 2 (Obstacles Layer)
        r_number_input = 1        
        ring_indexes_input = range(r_number_input * self.units_per_ring, (r_number_input + 1) * self.units_per_ring)
        # Add second input
        self.DirectConnection(ring_indexes, ring_indexes_input, -3)
        
        # Define self connection of the layer
        #setpoint_pattern = -1*np.array([1, 1, 1, 1, 1, -0.5, -1, -0.5, 1, 1, 1, 1, 1])
        
        setpoint_pattern = 1 * np.vstack((-0.5 * np.ones((int(self.units_per_ring / 2) - 1, 1)), np.array([[-0.2], [-0.5], [-0.2]]), -0.5 * np.ones((int(self.units_per_ring / 2) - 1, 1))))
        self.RingSelfConnections(setpoint_pattern.squeeze() , ring_indexes)

        # Current Orientation Layer ---------------------------------------------------------------------------------
        # Inputs for current orientation ring
        r_number = 3
        ring_indexes = range(r_number * self.units_per_ring, (r_number + 1) * self.units_per_ring)
        preferred_direction = np.linspace(-np.pi, np.pi, self.units_per_ring, endpoint = False)
        self.PreferentialOrientation(ring_indexes, preferred_direction, self.current_orientation)        
        current_pattern = np.vstack((-np.ones((int(self.units_per_ring / 2) - 1, 1)), np.array([[-0.5], [-1], [-0.5]]), -np.ones((int(self.units_per_ring / 2) - 1, 1))))
        self.RingSelfConnections(current_pattern.squeeze(), ring_indexes)
        

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
        cur_set_pattern = np.vstack((np.zeros((int(self.units_per_ring / 2), 1)) , -np.ones((int(self.units_per_ring / 2) - 1, 1))))
        self.RingOutConnections(cur_set_pattern.squeeze(), ring_indexes, ring_indexes_input)


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
        set_cur_pattern =  np.vstack((np.zeros((int(self.units_per_ring / 2), 1)) , -np.ones((int(self.units_per_ring / 2) - 1, 1))))
        self.RingOutConnections(set_cur_pattern.squeeze(), ring_indexes, ring_indexes_input)

        # Tank 1 ---------------------------------------------------------------------------------
        # Define input from Current minus Setpoint Layer
        r_number_input = 4
        ring_indexes_input = range(r_number_input * self.units_per_ring, (r_number_input + 1) * self.units_per_ring)
        r_number = 5
        unit_number = 0
        tank_index = (r_number + 1) * self.units_per_ring + unit_number
        self.TankConnections(np.ones((self.units_per_ring, 1)), tank_index, ring_indexes_input)

        # Tank 2 ---------------------------------------------------------------------------------
        # Define input from Setpoint minus Current Layer
        r_number_input = 5        
        ring_indexes_input = range(r_number_input * self.units_per_ring, (r_number_input + 1) * self.units_per_ring)
        r_number = 5
        unit_number = 1
        tank_index = (r_number + 1) * self.units_per_ring + unit_number
        self.TankConnections(np.ones((self.units_per_ring, 1)), tank_index, ring_indexes_input)

        # Forward vs Directional movement ---------------------------------------------------------------------------------

        # Out 1 unit: Tank 1 minus Tank 2

        r_number = 5
        unit_number = 2
        tank_index = (r_number + 1) * self.units_per_ring + unit_number

        # Define index of input Tank 1 to the unit
        r_number = 5
        unit_number = 0
        ring_indexes_input = (r_number + 1) * self.units_per_ring + unit_number

        self.DirectConnection(tank_index, ring_indexes_input, 1)

        # Define index of input Tank 2 to the unit
        r_number = 5
        unit_number = 1
        ring_indexes_input = (r_number + 1) * self.units_per_ring + unit_number

        self.DirectConnection(tank_index, ring_indexes_input, -1)


        # Out 2 unit: Tank 2 minus Tank 1

        r_number = 5
        unit_number = 3
        tank_index = (r_number + 1) * self.units_per_ring + unit_number

        # Define index of input Tank 1 to the unit
        r_number = 5
        unit_number = 0
        ring_indexes_input = (r_number + 1) * self.units_per_ring + unit_number

        self.DirectConnection(tank_index, ring_indexes_input, -1)

        # Define index of input Tank 2 to the unit
        r_number = 5
        unit_number = 1
        ring_indexes_input = (r_number + 1) * self.units_per_ring + unit_number

        self.DirectConnection(tank_index, ring_indexes_input, 1)


        # Speed output unit
        r_number = 5
        unit_number = 4
        tank_index = (r_number + 1) * self.units_per_ring + unit_number

        # Define input distance to target
        self.SetStim(tank_index, self.distance_to_target)

        # Define index of input Out 1 to the unit
        r_number = 5
        unit_number = 2
        ring_indexes_input = (r_number + 1) * self.units_per_ring + unit_number

        self.DirectConnection(tank_index, ring_indexes_input, -0.75)

        # Define index of input Out 2 to the unit
        r_number = 5
        unit_number = 3
        ring_indexes_input = (r_number + 1) * self.units_per_ring + unit_number

        self.DirectConnection(tank_index, ring_indexes_input, -0.75)

        # Define indices input from the obstacles layer
        r_number = 1        
        ring_indexes_obstacles = range(r_number * self.units_per_ring, (r_number + 1) * self.units_per_ring)


        # Define indices input from the current orientation layer
        r_number = 3
        ring_indexes_current = range(r_number * self.units_per_ring, (r_number + 1) * self.units_per_ring)

        self.ObstacleSpeedFactor(tank_index, ring_indexes_obstacles, ring_indexes_current,-0.5, 1, -0.3)

    def ObstacleSpeedFactor(self, ring_index, input_index_obstacles, input_index_current, w1, w2, w):
        product = np.expand_dims(w1 * self.current_firing_rate[input_index_obstacles], axis = 1) * np.expand_dims(w2 * self.current_firing_rate[input_index_current], axis = 1)
        product = np.squeeze(np.sum(product) / 24)
        self.presynaptic_potential[ring_index] += w * product   
    
    def DirectConnection(self, target_neuron_index, source_neuron_index, source_to_target_weight):        
        '''
            target_neuron_index: Index of the receiving neuron
            source_neuron_index: Index of the sending neuron
            source_to_target_weight: Weight of the connection
        '''
        self.presynaptic_potential[target_neuron_index] += source_to_target_weight * self.current_firing_rate[source_neuron_index]        
    
    def UpdateFunction(self, time, current_firing_rate):
        
        self.SetConnections()
        postsynaptic_potential = self.CalculatePostSynapticPotential(self.presynaptic_potential)
        activity_update = self.CalculateActivityUpdate(current_firing_rate, postsynaptic_potential)
        self.presynaptic_potential = np.zeros((self.n_neurons, 1))
        
        return activity_update
    
    def UpdateStates(self, new_states):
        self.current_firing_rate = new_states
        self.UpdateCurrentOrientation(0.001, 0.001)
        r_number_setpoint = 2
        ring_indexes_setpoint = range(r_number_setpoint * self.units_per_ring, (r_number_setpoint + 1) * self.units_per_ring)
        r_number_current = 3
        ring_indexes_current = range(r_number_current * self.units_per_ring, (r_number_current + 1) * self.units_per_ring)
        
    def RingSelfConnections(self, weights, ring_index):        
        inputs = self.current_firing_rate[ring_index, :]
        inputs_stacked = np.hstack((inputs.T, inputs.T, inputs.T))        
        n = len(weights)
        conv = np.convolve(np.flip(weights), inputs_stacked.squeeze())    
        conv = conv[int(np.fix(n / 2) + len(inputs)): int(np.fix(n / 2) + 2 * len(inputs))]        
        self.presynaptic_potential[ring_index, :] += np.expand_dims(conv, axis=1)
        
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
        r_number = 5
        tank_index_1 = (r_number + 1) * self.units_per_ring
        tank_index_2 = (r_number + 1) * self.units_per_ring + 1
        self.current_orientation += (-alpha*self.current_firing_rate[tank_index_1,:][0] + beta*self.current_firing_rate[tank_index_2,:][0])
        if self.current_orientation < 0:
            self.current_orientation += 360            
        if self.current_orientation > 360:
            self.current_orientation -= 360            
        print("Output sum", (-alpha*self.current_firing_rate[tank_index_1,:][0] + beta*self.current_firing_rate[tank_index_2,:][0]))
        print("Current Orientation", self.current_orientation)        
        
    def PlotRingActivity(self, index_ring, ring_firing_rates, fig_num, title):
        ring_activity = np.array(ring_firing_rates)[:,index_ring]
        #t = np.array(integrator_object.time_values)
        plt.figure(num = fig_num)
        plt.title(title)
        plt.plot(ring_activity.squeeze())
        plt.legend(np.round(np.degrees(np.linspace(-np.pi, np.pi, self.units_per_ring, endpoint = False))))        
        
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
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        # Create an array of angles for the circles to be plotted at
        angles = np.linspace(-180, 180, len(index_ring), endpoint = False)
        # Plot the circles
        for angle, sat in zip(angles, (((self.current_firing_rate[index_ring]) / np.amax(self.current_firing_rate[index_ring]))).tolist()):
            x_angle_text = 26 * np.cos(np.deg2rad(angle))
            y_angle_text = 26 * np.sin(np.deg2rad(angle))
            x = 20 * np.cos(np.deg2rad(angle))
            y = 20 * np.sin(np.deg2rad(angle))            
            circle = plt.Circle((x, y), radius=2.5, facecolor = (0.3, 0.4, 0.6, sat[0]), edgecolor="white", linewidth = 2)
            text = plt.text(x_angle_text - 1, y_angle_text - 1, str(int(angle)), fontsize = 9, color = (1, 1, 1))            
            ax.add_artist(circle)
            ax.add_artist(text)
        fig.set_size_inches((8, 8))
        ax.set_facecolor((0,0,0))

    def current_setpoint_circles(self, current_orientation_index_ring, setpoint_orientation_index_ring):
        # Create a figure and axes
        fig, ax = plt.subplots()
        # Set the axis limits to be square
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        # Create an array of angles for the circles to be plotted at
        angles = np.linspace(-180, 180, len(current_orientation_index_ring), endpoint = False)
        # Plot the circles
        for angle, sat in zip(angles, (((self.current_firing_rate[setpoint_orientation_index_ring]) / np.amax(self.current_firing_rate[setpoint_orientation_index_ring]))).tolist()):
            x_angle_text = 26 * np.cos(np.deg2rad(angle))
            y_angle_text = 26 * np.sin(np.deg2rad(angle))
            x = 20 * np.cos(np.deg2rad(angle))
            y = 20 * np.sin(np.deg2rad(angle))            
            circle = plt.Circle((x, y), radius=1.6, facecolor = (0.3, 0.4, 0.6, sat[0]), edgecolor="white", linewidth = 2)
            text = plt.text(x_angle_text - 1, y_angle_text - 1, str(int(angle)), fontsize = 9, color = (1, 1, 1))            
            ax.add_artist(circle)
            ax.add_artist(text)
        # Plot the circles
        for angle, sat in zip(angles, (((self.current_firing_rate[current_orientation_index_ring]) / np.amax(self.current_firing_rate[current_orientation_index_ring]))).tolist()):
            x = 14 * np.cos(np.deg2rad(angle))
            y = 14 * np.sin(np.deg2rad(angle))            
            circle = plt.Circle((x, y), radius=1.6, facecolor = (0.7, 0.4, 0.2, sat[0]), edgecolor="white", linewidth = 2)   
            ax.add_artist(circle)
            ax.add_artist(text)
        fig.set_size_inches((8, 8))
        ax.set_facecolor((0,0,0))

        

        
   
        

                
        