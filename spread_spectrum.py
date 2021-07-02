import numpy as np
import IPython
import matplotlib.pyplot as plt
from sequences import pnsequence
from scipy import signal

N13_barker_sequence = [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1]



params = {}

params['fs'] =1e6
params['sim_t'] = 10e-3
params['fc'] = 0.0078e6;
params['ts'] = 1 / params['fs']
params['tb'] = 1e-3


params['Nb'] = int(params['sim_t'] / params['tb'])
params['L'] = int(params['tb'] / params['ts'])
params['N'] = params['Nb'] * params['L']



def gen_barker_code(N):
	reps = int(np.ceil(N/len(N13_barker_sequence)))
	out = np.tile(N13_barker_sequence, reps)
	return out[0:N]

def generate_baseband(chips, params):

	baseband = np.array([-1 if c==0 else 1 for c in chips])

	baseband_upsample = np.zeros(len(chips) * params['L'])
	L = params['L']

	for i in range(0, len(chips)):
		baseband_upsample[(L*i):L*(i+1)] = baseband[i]

	return baseband_upsample

def generate_pn11_from_params(params):
	pn11 = np.zeros(params['N'])

	chips = pnsequence(11, '00110000100', [1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0], params['Nb'])
	baseband = generate_baseband(chips, params)
	pn11[0:len(baseband)] = baseband
	return pn11

def generate_carrier(params):

	n = np.arange(0, params['N'])
	carrier = np.cos(2 * np.pi * params['fc'] * n * params['ts'])
	
	return carrier


if __name__ == '__main__':

	# x = gen_barker_code(4096)
	pn_seed = np.zeros(11)
	pn_seed[3] = 1
	pn_mask = np.ones(11)
	seq_length = int(1e5)
	# x = pnsequence(11, pn_seed, pn_mask, seq_length)

	# x = pnsequence(11, '00110000100', [1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0], seq_length)

	# y = np.roll(x, 5)
	# z = np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(y)))

	x = pnsequence(11, '00110000100', [1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0], params['Nb'])

	baseband = generate_baseband(x, params)
	carrier = generate_carrier(params)
	bpsk = np.multiply(carrier, baseband)

	corr = carrier * bpsk

	# low pass filter
	nyquist = params['fs'] / 2
	cutoff = params['fc'] * 0.1
	cutoff_norm =  cutoff / nyquist
	b, a = signal.butter(3, cutoff_norm, btype='low')

	# x = np.reshpe(corr, params)

	corr_filter = signal.lfilter(b, a, corr)

	# ind = np.zeros(corr.shape)
	# curr = 0
	# for i in np.arange(params['N']):
	# 	curr = curr + corr(i)
	# 	if i == 

	ind_data = np.reshape(corr, (params['Nb'], int(len(corr)/params['Nb'])))
	ind_output1 = np.cumsum(ind_data, 1)
	ind_output = np.reshape(ind_output1, corr.shape)


	# # plt.plot(abs(np.fft.fft(carrier)))
	# # plt.plot(abs(np.fft.fft(bpsk)))

	# bpsk_autocorr = np.fft.ifft(np.multiply(np.fft.fft(bpsk), np.conj(np.fft.fft(bpsk))))
	# baseband_autocorr = np.fft.ifft(np.multiply(np.fft.fft(baseband), np.conj(np.fft.fft(baseband))))

	plt.plot(bpsk, '+-')
	plt.plot(carrier, '+-')
	plt.plot(corr, '+-')
	plt.plot(ind_output / max(ind_output), '+-')
	# plt.plot(corr_filter, '+-')
	


	IPython.embed()
