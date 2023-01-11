
import numpy as np
from matplotlib import pyplot as plt


tau_current = 1.5
tau_setpoint = 1.5
tau_cur_m_set = 1.5
tau_set_m_cur = 1.5
tau_left_tank = 0.2
tau_right_tank = 0.2

u = 12

input_angle = 90
current_angle = 0

angle_step = int(360/u)

preferred_direction = np.linspace(-np.pi, np.pi, u, endpoint = False)

h = 0.01

synapses11 = 1*np.ones((u))
synapses12 = 1*np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
synapses21 = 1*np.ones((u))
synapses22 = 1*np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

prev_tank = 1
prev_tank_1 = 0
tanks_syn = 0


def circ_conv(fltr, inputs):
    n = len(fltr)
    conv = np.convolve(np.flip(fltr), np.hstack((inputs, inputs, inputs)))

    conv = conv[int(np.fix(n / 2) + len(inputs)): int(np.fix(n / 2) + 2 * len(inputs))]
    return conv

def Neuron_func(Neuron_func_val, Input, M, S, A, tau):
	return (1/tau)*(- Neuron_func_val + NR(Input, M, S, A))


def NR(X, M, S, A):
	return M * ((X * (X > 0)) ** 2)/ ((S + A )** 2 + ((X * (X > 0)) ** 2))


setpoint_list = []
current_status_list = []
cur_m_set_list = []
set_m_cur_list = []
left_tank_list = []
right_tank_list = []
    
n = 2
distance = 100

setpoint = np.zeros((u, n), dtype = np.float64)
current_status = np.zeros((u, n), dtype = np.float64)
cur_m_set = np.zeros((u, n), dtype = np.float64)
set_m_cur = np.zeros((u, n), dtype = np.float64)
left_tank = np.zeros((1, n), dtype = np.float64)
right_tank = np.zeros((1, n), dtype = np.float64)

count = 0

while count < 4000:
    

    input_setpoint = distance*np.cos(preferred_direction - np.radians(input_angle))*(distance*np.cos(preferred_direction - np.radians(input_angle)) >= 0)
    input_current = distance*np.cos(preferred_direction - np.radians(current_angle))*(distance*np.cos(preferred_direction - np.radians(current_angle)) >=0)

    

    k1_setpoint =       h * Neuron_func(setpoint[:, 0], input_setpoint, 100, 50, 0, tau_setpoint)
    k1_current_status = h * Neuron_func(current_status[:, 0], input_current, 100, 50, 0, tau_current)

    k1_cur_m_set =      h * Neuron_func(cur_m_set[:, 0], synapses11*current_status[:, 0] - circ_conv(synapses12,setpoint[:, 0]), 100, 50, 0, tau_cur_m_set)
    k1_set_m_cur =      h * Neuron_func(set_m_cur[:, 0], synapses21*setpoint[:, 0] - circ_conv(synapses22,current_status[:, 0]), 100, 50, 0, tau_set_m_cur)

    k1_left_tank =      h * Neuron_func(left_tank[:, 0], prev_tank*np.sum(cur_m_set[:, 0]) - prev_tank_1*np.sum(set_m_cur[:, 0]) - tanks_syn*right_tank[:, 0], 100, 50, 0, tau_left_tank)
    k1_right_tank =     h * Neuron_func(right_tank[:, 0], prev_tank*np.sum(set_m_cur[:, 0]) - prev_tank_1*np.sum(cur_m_set[:, 0]) - tanks_syn*left_tank[:, 0], 100, 50, 0, tau_right_tank)

    k2_setpoint =       h * Neuron_func(setpoint[:, 0] + k1_setpoint*h/2, input_setpoint, 100, 50, 0, tau_setpoint)
    k2_current_status = h * Neuron_func(current_status[:, 0] + k1_current_status*h/2, input_current, 100, 50, 0, tau_current)

    k2_cur_m_set =      h * Neuron_func(cur_m_set[:, 0] + k1_cur_m_set*h/2, synapses11*current_status[:, 0] - circ_conv(synapses12,(setpoint[:, 0])), 100, 50, 0, tau_cur_m_set)
    k2_set_m_cur =      h * Neuron_func(set_m_cur[:, 0] + k1_set_m_cur*h/2, synapses21*(setpoint[:, 0] ) - circ_conv(synapses22,(current_status[:, 0] )), 100, 50, 0, tau_set_m_cur)

    k2_left_tank =      h * Neuron_func(left_tank[:, 0] + k1_left_tank*h/2, prev_tank*np.sum(cur_m_set[:, 0] ) - prev_tank_1*np.sum(set_m_cur[:, 0] ) - tanks_syn*(right_tank[:, 0] ), 100, 50, 0, tau_left_tank)
    k2_right_tank =     h * Neuron_func(right_tank[:, 0] + k1_right_tank*h/2, prev_tank*np.sum(set_m_cur[:, 0] ) - prev_tank_1*np.sum(cur_m_set[:, 0] ) - tanks_syn*(left_tank[:, 0] ), 100, 50, 0, tau_right_tank)


    k3_setpoint =       h * Neuron_func(setpoint[:, 0] + k2_setpoint*h/2, input_setpoint, 100, 50, 0, tau_setpoint)
    k3_current_status = h * Neuron_func(current_status[:, 0] + k2_current_status*h/2, input_current, 100, 50, 0, tau_current)

    k3_cur_m_set =      h * Neuron_func(cur_m_set[:, 0] + k2_cur_m_set*h/2, synapses11*(current_status[:, 0]) - circ_conv(synapses12,(setpoint[:, 0] )), 100, 50, 0, tau_cur_m_set)
    k3_set_m_cur =      h * Neuron_func(set_m_cur[:, 0] + k2_set_m_cur*h/2, synapses21*(setpoint[:, 0]) - circ_conv(synapses22,(current_status[:, 0] )), 100, 50, 0, tau_set_m_cur)

    k3_left_tank =      h * Neuron_func(left_tank[:, 0] + k2_left_tank*h/2, prev_tank*np.sum(cur_m_set[:, 0] ) - prev_tank_1*np.sum(set_m_cur[:, 0] ) - tanks_syn*(right_tank[:, 0] ), 100, 50, 0, tau_left_tank)
    k3_right_tank =     h * Neuron_func(right_tank[:, 0] + k2_right_tank*h/2, prev_tank*np.sum(set_m_cur[:, 0] ) - prev_tank_1*np.sum(cur_m_set[:, 0] ) - tanks_syn*(left_tank[:, 0] ), 100, 50, 0, tau_right_tank)


    k4_setpoint =       h * Neuron_func(setpoint[:, 0] + k3_setpoint*h, input_setpoint, 100, 50, 0, tau_setpoint)
    k4_current_status = h * Neuron_func(current_status[:, 0] + k3_current_status*h, input_current, 100, 50, 0, tau_current)

    k4_cur_m_set =      h * Neuron_func(cur_m_set[:, 0] + k3_cur_m_set*h, synapses11*(current_status[:, 0] ) - circ_conv(synapses12,(setpoint[:, 0] )), 100, 50, 0, tau_cur_m_set)
    k4_set_m_cur =      h * Neuron_func(set_m_cur[:, 0] + k3_set_m_cur*h, synapses21*(setpoint[:, 0] ) - circ_conv(synapses22,(current_status[:, 0] )), 100, 50, 0, tau_set_m_cur)

    k4_left_tank =      h * Neuron_func(left_tank[:, 0] + k3_left_tank*h, prev_tank*np.sum(cur_m_set[:, 0] ) - prev_tank_1*np.sum(set_m_cur[:, 0] ) - tanks_syn*(right_tank[:, 0] ), 100, 50, 0, tau_left_tank)
    k4_right_tank =     h * Neuron_func(right_tank[:, 0] + k3_right_tank*h, prev_tank*np.sum(set_m_cur[:, 0] ) - prev_tank_1*np.sum(cur_m_set[:, 0] ) - tanks_syn*(left_tank[:, 0] ), 100, 50, 0, tau_right_tank)



    A       = setpoint[:, 0]       + (1/6)*(k1_setpoint       + 2 * k2_setpoint       + 2 * k3_setpoint       + k4_setpoint)
    B = current_status[:, 0] + (1/6)*(k1_current_status + 2 * k2_current_status + 2 * k3_current_status + k4_current_status)
    C      = cur_m_set[:, 0]      + (1/6)*(k1_cur_m_set      + 2 * k2_cur_m_set      + 2 * k3_cur_m_set      + k4_cur_m_set)
    D      = set_m_cur[:, 0]      + (1/6)*(k1_set_m_cur      + 2 * k2_set_m_cur      + 2 * k3_set_m_cur      + k4_set_m_cur)
    E      = left_tank[:, 0]      + (1/6)*(k1_left_tank      + 2 * k2_left_tank      + 2 * k3_left_tank      + k4_left_tank)
    F     = right_tank[:, 0]     + (1/6)*(k1_right_tank     + 2 * k2_right_tank     + 2 * k3_right_tank     + k4_right_tank)
    
    alpha = 0.001
    beta = 0.001
    
    current_angle += (alpha*F - beta*E)
    if current_angle < 0:
        current_angle += 360
        
    if current_angle > 360:
        current_angle -= 360
    
    
    
    setpoint[:, 1]       = A
    current_status[:, 1] = B
    cur_m_set[:, 1]      = C
    set_m_cur[:, 1]      = D
    left_tank[:, 1]      = E
    right_tank[:, 1]     = F

    

    setpoint[:, 0]       = setpoint[:, 1]
    current_status[:, 0] = current_status[:, 1]
    cur_m_set[:, 0]      = cur_m_set[:, 1]
    set_m_cur[:, 0]      = set_m_cur[:, 1]
    left_tank[:, 0]      = left_tank[:, 1]
    right_tank[:, 0]     = right_tank[:, 1]
    
    
    setpoint_list.append(A)
    current_status_list.append(B)
    cur_m_set_list.append(C)
    set_m_cur_list.append(D)
    left_tank_list.append(E)
    right_tank_list.append(F)
    
    count += 1

plt.figure()
plt.plot(setpoint_list)
plt.legend(np.round(np.degrees(np.linspace(-np.pi, np.pi, 12, endpoint = False))))

plt.figure()
plt.plot(current_status_list)
plt.legend(np.round(np.degrees(np.linspace(-np.pi, np.pi, 12, endpoint = False))))

plt.figure()
plt.plot(cur_m_set_list)
plt.legend(np.round(np.degrees(np.linspace(-np.pi, np.pi, 12, endpoint = False))))

plt.figure()
plt.plot(set_m_cur_list)
plt.legend(np.round(np.degrees(np.linspace(-np.pi, np.pi, 12, endpoint = False))))

plt.figure()
plt.plot(left_tank_list)
plt.legend(np.round(np.degrees(np.linspace(-np.pi, np.pi, 12, endpoint = False))))

plt.figure()
plt.plot(right_tank_list)
plt.legend(np.round(np.degrees(np.linspace(-np.pi, np.pi, 12, endpoint = False))))

plt.show()

print(current_angle)

        


