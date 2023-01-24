# BINNF - Bio-Inspired Neural Network Framework

This project is looking for creating a framework for easily creating and simulating Bio-Inspired Neural Networks based on dynamical systems. Currently this framework supports creating networks whose cells/units follow the mean firing rate concept. In the future it should be possible to simulate other kind of cells/units.

Also, currently this framework uses the Runge-Kutta 4 method for integrating the differential equations. 

# How to execute
The example we are showing here is a new neural network that will be fully explain when future updates come. For now, if you want to test the working of this integrator follow these steps:

Create a virtual environment in anaconda or venv and install the required libraries:

```sh 
pip install matplotlib numpy
```

Clone or download the repository:
```sh 
git clone https://github.com/JesusRiveroOrtega/BINNF.git
```

Execute the example:
```sh
python RK4_Integration.py
```



