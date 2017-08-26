import bernardLindner as bl
import numpy as np 
from numpy.fft import fft
import scipy.signal as spy
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import glob
import time
'''
def simulation(tsim, dt):

	tsim = tsim
	dt = dt

	spikeTrain, S_t = bl.poissonNeuron(r0 = 100, dt = dt, epslon = 0.2, tsim = tsim)
	#spikeTrain, S_t = bl.stochasticLIF(tsim = tsim, c = 0.98, D = 3.3, tau_m = 10, dt = dt, mu = 0.75, tabs = 0.0)
	#spikeTrain, S_t = bl.EIFmodel(tau_m = 10, mu = 0.7, D = 3.3, c = 0.98, dt = dt, tsim = tsim, delta_t = 0.1, tabs = 0.0)
	#bl.plotSpikeTrain(spikeTrain, dt)
	f, ilb , mir = bl.MRIfrequency(spikeTrain, S_t, dt)
	return mir


s = simulation(2000, 1.0)
'''

def rf():
	np.random.seed(seed=int(time.time()))
	return int(abs(np.random.rand()*10))


Ntrials = int(5e6)

dt = 1.0
Tsim = 50.0

words = []
possibleWords = 2**int(Tsim/dt)

def genWords(ss, sn):
	#spikeTrain, S_t = bl.stochasticLIF(seed_signal = ss, seed_noise = sn, tsim = Tsim, c = 0.98, D = 3.3, tau_m = 10, dt = dt, mu = 0.75, tabs = 0.0)
	spikeTrain, S_t = bl.poissonNeuron(seed_signal = ss, seed_noise = sn, r0 = 100, dt = dt, epslon = 0.8, tsim = Tsim)
	word = bl.st2word(spikeTrain)
	return word

r''' 
	Dynamic Stimulus
'''
num_cores = multiprocessing.cpu_count()
words = Parallel(n_jobs=num_cores)(delayed(genWords)((Ntrials-i), rf()*i) for i in range(0, Ntrials))

data_frame = pd.DataFrame(words, columns=['words'])
count = data_frame.groupby(['words'])['words'].count()

protocol_1 = pd.DataFrame(count.index, columns = ['words'])
protocol_1.insert(1, 'p_x', 1.0 * count.values / sum(count.values))

Hx = -sum(protocol_1['p_x']*np.log2(protocol_1['p_x']))

print Hx 

r''' 
	Frozen Signal: Uses an ensemble of 10 different signals
'''
signal = range(0,10)
Hxy = 0
for s in signal:

	num_cores = multiprocessing.cpu_count()
	words2 = Parallel(n_jobs=num_cores)(delayed(genWords)(s, rf()*i) for i in range(0, Ntrials))

	data_frame2 = pd.DataFrame(words2, columns=['words'])
	count2 = data_frame2.groupby(['words'])['words'].count()

	protocol_2 = pd.DataFrame(count2.index, columns = ['words'])
	protocol_2.insert(1, 'p_xy', 1.0 * count2.values / sum(count2.values))
	print -sum(protocol_2['p_xy']*np.log2(protocol_2['p_xy']))
	Hxy = Hxy - sum(protocol_2['p_xy']*np.log2(protocol_2['p_xy']))

Hxy = Hxy*1.0 / len(signal)
print Hxy