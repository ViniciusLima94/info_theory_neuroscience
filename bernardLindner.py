r'''
	Neuron Models as described in Bernardi & Lindiner 2014.
'''

import numpy as np 
import random
import matplotlib.pyplot as plt
import scipy.signal as spy
import frozenSignal as fs

r'''
	Simulate a poisson neuron.
	r0 : mean firing rate in Hz
	dt : time step in ms
	epslon : Strength Factor
	tsim : simulation time in s
	returns the spike train series and the signal S_t
'''
def poissonNeuron(seed_signal = 1, seed_noise = 2, r0 = 100, dt = 1.0, epslon = 1.0, tsim = 2000):

	dt     = dt       # Resolution [ms]
	epslon = epslon   # Sinal Strength
	r0     = r0       # mean firing rate [Hz]

	tsim   = tsim  # Simulation time [ms]
	time   = np.arange(0, tsim, dt)            # Time vector
	spike_train =  np.zeros(len(time))         # Spike train vector

	np.random.seed(seed = seed_signal)
	S_t         =  np.random.randn(len(time))  # Signal S(t), gaussian with mean zero and std one

	np.random.seed(seed = seed_noise)
	for i in range(0, len(time)):
		r_t = r0*(1 + epslon*S_t[i])
		if  np.random.rand() < r_t*dt*1e-3:
			spike_train[i] = 1.0 / dt

	return spike_train, S_t

r'''
	Stochastic LIF neuron.
	tau_m : membrane time constant in ms
	dt : time step in ms
	mu    : rest potential in mV
	D     : overall noise intensity
	c     : relative strength of the signal
	tabs  : refractory period in ms
'''
def stochasticLIF(seed_signal = 1, seed_noise = 2, tau_m = 10.0, mu = 0.5, D = 5, c = 0.8, dt = 0.1, tsim = 2000, tabs = 2.0):
	
	dt     = dt       # Resolution [ms]

	tsim   = tsim  # Simulation time [ms]
	time   = np.arange(0, tsim, dt)            # Time vector
	v      = np.zeros(len(time))               # Membrane potential vector
	spike_train =  np.zeros(len(time))         # Spike train vector

	np.random.seed(seed = seed_signal)
	S_t    = np.random.randn(len(time))  # Signal S(t), gaussian with mean zero and std one

	np.random.seed(seed = seed_noise)
	N_t    = np.random.randn(len(time))  # Signal S(t), gaussian with mean zero and std one

	v[0]   = np.random.rand()            # Membrane potential initial value

	vt, vr     = 1.0, 0.0
	X = 0
	tspike = 0
	for i in range(0, len(time)-1) :
		if X == 1:
			v[i+1] = vr
			if time[i] - tspike >= tabs:
				X = 0
		else:
			v[i+1] = v[i] + (dt/tau_m) * ( -v[i] + mu + np.sqrt(2*D*c)*S_t[i] + np.sqrt(2*D*(1-c))*N_t[i] )
			if v[i+1] > vt:
				spike_train[i] = 1.0 / dt
				tspike         = time[i]   
				v[i+1]         = vr

	return spike_train, S_t
	

r'''
	Stochastic EIF neuron.
	tau_m : membrane time constant in ms
	dt : time step in ms
	mu    : rest potential in mV
	D     : overall noise intensity
	c     : relative strength of the signal
	delta_t : steepness of the signal ms
	tabs  : refractory period in ms
'''
def EIFmodel(seed_signal = 1, seed_noise = 2, tau_m = 10, mu = 0.5, D = 10, c = 0.8, dt = 0.1, tsim = 2000, delta_t = 0.1, tabs = 2.0):
	dt     = dt       # Resolution [ms]

	tsim   = tsim  # Simulation time [ms]
	time   = np.arange(0, tsim, dt)            # Time vector
	v      = np.zeros(len(time))               # Membrane potential vector
	spike_train =  np.zeros(len(time))         # Spike train vector

	np.random.seed(seed = seed_signal)
	S_t    = np.random.randn(len(time))  # Signal S(t), gaussian with mean zero and std one

	np.random.seed(seed = seed_noise)
	N_t    = np.random.randn(len(time))  # Signal S(t), gaussian with mean zero and std one

	v[0]   = np.random.rand()            # Membrane potential initial value

	vt, vr     = 2.0, 0.0
	X = 0
	tspike = 0
	for i in range(0, len(time)-1) :
		if X == 1:
			v[i+1] = vr
			if time[i] - tspike >= tabs:
				X = 0
		else:
			v[i+1] = v[i] + (dt/tau_m) * ( -v[i] + delta_t*np.exp((v[i]-1)/delta_t) + mu + np.sqrt(2*D*c)*S_t[i] + np.sqrt(2*D*(1-c))*N_t[i] )
			if v[i+1] > vt:
				spike_train[i] = 1.0 / dt
				tspike         = time[i]   
				v[i+1]         = vr

	return spike_train, S_t

r'''	
	Plot a Spike Train series
	spike_train : spike train to be ploted
	dt          : resolution of bins in ms
'''
def plotSpikeTrain(spike_train, dt):
	time = np.squeeze(np.where(spike_train == 1))
	time = time.astype(float) * dt
	[plt.plot([time[i], time[i]], [0, 1], 'blue') for i in range(0, len(time))]
	plt.show()

r'''
	Converts a spike train to a string of 0's and '1's
'''
def st2word(spike_train):
	word = ''
	for i in range(0, len(spike_train)):
		word = word + str( spike_train[i].astype(int) )
	return word

r'''
	Calculate MRI frequency dependent.
	spike_train : neuron spike train
	signal      : gaussian signal 
	dt          : integration step in ms
	returns frequency, Ilb and MIR
'''
def MRIfrequency(spike_train, signal, dt):

	f, cxy = spy.coherence(spike_train, signal, fs = 1.0/dt)

	ilb = -np.log2(1-cxy)

	MIR = sum((f[1]-f[0])*ilb)

	return f, ilb, MIR