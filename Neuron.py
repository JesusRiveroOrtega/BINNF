# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 19:08:46 2022

@author: jrive
"""
import numpy as np
from matplotlib import pyplot as plt

class Neuron:
    def __init__(self, n_neurons, semisaturation, max_saturation, input_exp, tau, units_per_ring, setpoint_ring_connections):
        
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
        self.setpoint_orientation = 0
        self.units_per_ring = units_per_ring
        self.obstacle_orientations = None
        self.distance_to_target = 0
        self.setpoint_ring_connections = setpoint_ring_connections
        
        
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

    def SetTargetOrientation(self, new_target):     
        self.setpoint_orientation = new_target

    def SetDistance_to_Target(self, new_distance_to_target):     
        self.distance_to_target = new_distance_to_target

    def SetCurrentOrientation(self, new_current):
        self.current_orientation = new_current
    	
    def SetObstaclesOrientation(self, new_obstacle_orientations):
        self.obstacle_orientations = new_obstacle_orientations

    def SetStimObstacles(self, ring_index, orientations, distances):

        preferred_direction = np.linspace(-np.pi, np.pi, self.units_per_ring, endpoint = False)
        projection = np.zeros_like(preferred_direction)

        for orientation_idx in range(len(orientations)):
            projection_idx = np.argmin(np.abs(preferred_direction - orientations[orientation_idx]))
            projection[projection_idx] += (120 / distances[orientation_idx])

        projection[projection > 100] = 100        
        self.presynaptic_potential[ring_index, :] += np.expand_dims(projection, axis = 1)
    
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
        if self.setpoint_orientation == None:
            pass
        else:
            ring_indexes = range(r_number * self.units_per_ring, (r_number + 1) * self.units_per_ring)
            self.SetGaussianStim(ring_indexes, self.setpoint_orientation, 1.2, 5, 100, self.units_per_ring)

        # Obstacles Layer
        ## Ring Number = 1
        r_number = 1        
        ring_indexes = range(r_number * self.units_per_ring, (r_number + 1) * self.units_per_ring)        
        #self.SetStimObstacles(ring_indexes,  np.deg2rad(self.obstacle_orientations[0]), self.obstacle_orientations[1])
        if self.obstacle_orientations == None:
            pass
        else:
            self.SetGaussianStim(ring_indexes, self.obstacle_orientations[0][0], 0.4, 5, 100, self.units_per_ring)

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
        self.DirectConnection(ring_indexes, ring_indexes_input, -2)
        
        # Define self connection of the layer
        setpoint_pattern = 2*np.vstack((-0.95 * np.ones((int(self.units_per_ring / 2) - 1, 1)), 0.85 * np.ones((3, 1)), -0.95 * np.ones((int(self.units_per_ring / 2) - 1, 1))))
        #setpoint_pattern = np.vstack((-1.4 * np.ones((int(self.units_per_ring / 2) - 1, 1)), 1.0 * np.ones((3, 1)), -1.4 * np.ones((int(self.units_per_ring / 2) - 1, 1))))
        
        self.RingSelfConnections(self.setpoint_ring_connections.squeeze() , ring_indexes)
        #self.RingSelfConnections(setpoint_pattern.squeeze() , ring_indexes)

        # Current Orientation Layer ---------------------------------------------------------------------------------
        # Inputs for current orientation ring
        r_number = 3
        ring_indexes = range(r_number * self.units_per_ring, (r_number + 1) * self.units_per_ring)
        preferred_direction = np.linspace(-np.pi, np.pi, self.units_per_ring, endpoint = False)
        if self.current_orientation is None:
            pass        
        else:
            self.PreferentialOrientation(ring_indexes, preferred_direction, self.current_orientation)
        current_pattern = np.vstack((-0.9 * np.ones((int(self.units_per_ring / 2) - 1, 1)), 0.8 * np.ones((3, 1)), -0.9 * np.ones((int(self.units_per_ring / 2) - 1, 1))))
        self.RingSelfConnections(self.setpoint_ring_connections.squeeze() , ring_indexes)
        #self.RingSelfConnections(current_pattern.squeeze(), ring_indexes)
        

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

        if self.distance_to_target == None:
            pass
        else:
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
        #self.UpdateCurrentOrientation(0.0005, 0.0005)
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
        #print("Output sum", (-alpha*self.current_firing_rate[tank_index_1,:][0] + beta*self.current_firing_rate[tank_index_2,:][0]))
        #print("Current Orientation", self.current_orientation)        
        
    """ def PlotRingActivity(self, index_ring, ring_firing_rates, t, fig_num, title, sens_params):
        ring_activity = np.array(ring_firing_rates)[:,index_ring]
        
        plt.figure(num = fig_num)
        plt.suptitle(title + " Activity Level")
        subtitle = "Far connections: " + str(sens_params[0]) + " Near and Auto connections: " + str(sens_params[1])
        plt.title(subtitle)
        plt.ylim([0, 100])
        plt.ylabel("Mean Firing Rate (Hz)")
        plt.xlabel("Time (s)") 
        plt.plot(t, ring_activity.squeeze(), linewidth = 3)
        plt.legend(np.round(np.degrees(np.linspace(-np.pi, np.pi, self.units_per_ring, endpoint = False))))
            
        filename = "far_" + str(sens_params[0]) + "_near_auto_" + str(sens_params[1])
        plt.savefig("C:/Users/jrive/Documents/Projects/HeadingNet/sens9" + "/" + filename + ".png")
        plt.clf() """

    def PlotRingActivity(self, index_ring, ring_firing_rates, t, fig_num, title):
        ring_activity = np.array(ring_firing_rates)[:,index_ring]
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(t, ring_activity.squeeze(), linewidth = 5)
        ax.axvline(x = 0.5, ymin = 0, ymax = 100, color = "black", linewidth = 3, linestyle = "-")
        # ax.axvline(x = 1.0, ymin = 0, ymax = 100, color = "black", linewidth = 3, linestyle = ":")
        # ax.axvline(x = 1.5, ymin = 0, ymax = 100, color = "black", linewidth = 3, linestyle = "--")
        #ax.axvline(x = 2.0, ymin = 0, ymax = 100, color = "black", linewidth = 3, linestyle = "dashdot")
        ax.set_xlabel("Time (s)", fontsize = 24)
        ax.set_ylabel("Mean Firing Rate (Hz)", fontsize = 24)
        ax.set_title(title + " Activity Level", fontsize = 24)
        
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(np.linspace(-180, 180, self.units_per_ring, endpoint = False), loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 19)
        
                
    def PlotNeuronsActivity(self, index_neuron, neuron_firing_rates, fig_num, title):
        neuron_activity = np.array(neuron_firing_rates)[:,index_neuron]
        #t = np.array(integrator_object.time_values)
        plt.figure(num = fig_num)
        plt.title(title)
        plt.plot(neuron_activity.squeeze())
        
    def DecodeOrientation(self, activity, ring_index, n_neurons_ring):
        activity = np.array(activity).squeeze()[ring_index]
        orientations = np.linspace(-180, 180, n_neurons_ring, endpoint = False)
        return np.sum(activity * orientations) / np.sum(activity)
        
    def plot_circles(self, index_ring, specific_color):
        # Create a figure and axes
        fig, ax = plt.subplots()
        # Set the axis limits to be square
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        # Create an array of angles for the circles to be plotted at
        angles = np.linspace(-180, 180, len(index_ring), endpoint = False)
        activ = self.current_firing_rate[index_ring] / 120
        
        # Plot the circles
        for angle, sat in zip(angles, (activ / np.amax(activ)).tolist()):
            x_angle_text = 26.5 * np.cos(np.deg2rad(angle))
            y_angle_text = 26.5 * np.sin(np.deg2rad(angle))
            x = 20 * np.cos(np.deg2rad(angle))
            y = 20 * np.sin(np.deg2rad(angle))            
            circle = plt.Circle((x, y), radius=2.5, facecolor = (specific_color[0], specific_color[1], specific_color[2], sat[0]), edgecolor="black", linewidth = 2)
            text = plt.text(x_angle_text - 2, y_angle_text - 2, str(int(angle)), fontsize = 15, color = (0, 0, 0))            
            ax.add_artist(circle)
            ax.add_artist(text)
        fig.set_size_inches((8, 8))
        ax.set_facecolor((1,1,1))
        ax.axis("off")

    def current_setpoint_circles(self, current_orientation_index_ring, setpoint_orientation_index_ring, current_specific_color, setpoint_specific_color):
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
            circle = plt.Circle((x, y), radius=1.6, facecolor = (current_specific_color[0], current_specific_color[1], current_specific_color[2], sat[0]), edgecolor="black", linewidth = 2)
            text = plt.text(x_angle_text - 1, y_angle_text - 1, str(int(angle)), fontsize = 9, color = (0, 0, 0))            
            ax.add_artist(circle)
            ax.add_artist(text)
        # Plot the circles
        for angle, sat in zip(angles, (((self.current_firing_rate[current_orientation_index_ring]) / np.amax(self.current_firing_rate[current_orientation_index_ring]))).tolist()):
            x = 14 * np.cos(np.deg2rad(angle))
            y = 14 * np.sin(np.deg2rad(angle))            
            circle = plt.Circle((x, y), radius=1.6, facecolor = (setpoint_specific_color[0], setpoint_specific_color[1], setpoint_specific_color[2], sat[0]), edgecolor="black", linewidth = 2)   
            ax.add_artist(circle)
            ax.add_artist(text)
        fig.set_size_inches((8, 8))
        ax.set_facecolor((1,1,1))

    def PlotRingActivityPar(self, index_ring, ring_firing_rates, t, fig_num, title, sens_params, specific_color):
        ring_activity = np.array(ring_firing_rates)[:,index_ring]
        
        plt.clf()
        plt.rcParams.update({'font.size': 18})
        fig, (activity_ax, ring_ax) = plt.subplots(1, 2)
        activity_ax.plot(t, ring_activity.squeeze(), linewidth = 4)
        #activity_ax.axvline(x = 0.25, ymin = 0, ymax = 100, color = "black", linewidth = 2, linestyle = "--")      
        activity_ax.set_xlabel("Time (s)")
        activity_ax.set_ylabel("Mean Firing Rate (Hz)")
        activity_ax.set_title(title + " Activity Level" + "\n" + " Inhibitions " + str(sens_params[0]) + " Excitatory " + str(sens_params[1]))
        activity_ax.set_ylim(0, 100)
        # Shrink current axis by 20%
        box = activity_ax.get_position()
        activity_ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        activity_ax.legend(np.round(np.degrees(np.linspace(-np.pi, np.pi, self.units_per_ring, endpoint = False))), loc='center left', bbox_to_anchor=(1, 0.5))

        ring_ax.set_xlim(-30, 30)
        ring_ax.set_ylim(-30, 30)
        # Create an array of angles for the circles to be plotted at
        angles = np.linspace(-180, 180, len(index_ring), endpoint = False)
        activ = (ring_activity[-1, :].squeeze() / 100).tolist()
        
        # Plot the circles
        for angle, sat in zip(angles, activ):
            x_angle_text = 26 * np.cos(np.deg2rad(angle))
            y_angle_text = 26 * np.sin(np.deg2rad(angle))
            x = 20 * np.cos(np.deg2rad(angle))
            y = 20 * np.sin(np.deg2rad(angle))            
            circle = plt.Circle((x, y), radius=2.5, facecolor = (specific_color[0], specific_color[1], specific_color[2], sat), edgecolor="black", linewidth = 2)
            text = plt.text(x_angle_text - 1, y_angle_text - 1, str(int(angle)), fontsize = 14, color = (0, 0, 0))            
            ring_ax.add_artist(circle)
            ring_ax.add_artist(text)
        
        ring_ax.set_facecolor((1,1,1))
        ring_ax.set_aspect(1)
        ring_ax.axis("off")

        fig.tight_layout(pad=20)
        fig.set_size_inches(25, 10)
        filename = " far_" + '{0:.2f}'.format(sens_params[0]) + " near_auto_" + '{0:.2f}'.format(sens_params[1])
        plt.savefig("C:/Users/jrive/Documents/Projects/HeadingNet/obstacle" + "/" + title + "/" + filename + ".png")
        plt.close("all")

    def RegisterLims(self, index_ring, ring_firing_rates, sens_params, layer_name):
        print()
        ring_activity = np.array(ring_firing_rates)[:,index_ring]
        ring_activity = ring_activity[-1, :].squeeze()

        ring_activity_bin = (ring_activity > 1).tolist()
        inhibition_type = 0
        excitation_type = 0
        if sum(ring_activity_bin) < len(ring_activity.tolist()): #does inhibition make at least a neuron to die?
            inhibition_type = 1
        if sum(ring_activity_bin) == 3: #does inhibition make that only three units are active?
            inhibition_type = 2
        if sum(ring_activity_bin) == 0: #does inhibition make that all neurons die?
            inhibition_type = 3
        
        
        if sum(ring_activity_bin) > 0: #does excitation make that at least one neuron survives?
            excitation_type = 1
        if sum(ring_activity_bin) == 3: #does excitation make that only three units are active?
            excitation_type = 2
        
        

        from csv import writer

        file_line = [inhibition_type, sens_params[0], excitation_type, sens_params[1]] 
        # Open our existing CSV file in append mode
        # Create a file object for this file
        with open('obstacle/' + layer_name + 'params_limits.csv', 'a') as f_object:
        
            # Pass this file object to csv.writer()
            # and get a writer object
            writer_object = writer(f_object)
        
            # Pass the list as an argument into
            # the writerow()
            writer_object.writerow(file_line)
        
            # Close the file object
            f_object.close()
   
        

                
        