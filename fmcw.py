import numpy as np
import matplotlib.pyplot as plt
import IPython
from spread_spectrum import generate_pn11_from_params


def baseband_chirp_phase(B, N):
	n = np.arange(N)
	phase = (B/(2*(N-1))) * np.power(n, 2) - (B/2) * n
	return phase

def baseband_chirp(B, N):
	return np.exp(1j * baseband_chirp_phase(B, N))


def baseband_chirp_from_params(params):
	chirp = np.zeros(params['N'])
	chirp_nonzero = np.exp(1j * baseband_chirp_phase(params['chirp_B'] / params['fs'], params['Nchirp']))
	chirp[0:len(chirp_nonzero)] = chirp_nonzero
	return chirp


class RangeDetector(object):

	def __init__(self, ref_wform):
		self.ref = ref_wform
		self.ref_fft = np.fft.fft(self.ref)

	def calculate_raw(self, rx_wform):
		return np.full_like(rx_wform, 0)

class MatchedFilterDetector(RangeDetector):

	def calculate_raw(self, rx_wform):
		return np.fft.ifft(np.multiply(np.conj(self.ref_fft), np.fft.fft(rx_wform)))

class ClassicFMCWDetector(RangeDetector):

	def calculate_raw(self, rx_wform):
		return np.fft.fft(np.multiply(np.conj(self.ref), rx_wform))


if __name__ == '__main__':


	chirp_params = {}

	chirp_params['fs'] =1e6
	chirp_params['sim_t'] = 100e-3
	chirp_params['fc'] = 0.05e6;
	chirp_params['ts'] = 1 / chirp_params['fs']
	chirp_params['N'] = int(chirp_params['sim_t'] / chirp_params['ts'])

	chirp_params['chirp_s'] = chirp_params['ts'] * 8192
	chirp_params['chirp_B'] = 0.05e6
	chirp_params['Nchirp'] = int(chirp_params['chirp_s'] / chirp_params['ts'])



	pn11_params = {}

	pn11_params['fs'] =1e6
	pn11_params['sim_t'] = 100e-3
	pn11_params['ts'] = 1 / pn11_params['fs']
	pn11_params['tb'] = 100 * pn11_params['ts']
	pn11_params['pn11_s'] = pn11_params['ts'] * 8192
	pn11_params['N'] = int(pn11_params['sim_t'] / pn11_params['ts'])

	pn11_params['Nb'] = int(pn11_params['pn11_s'] / pn11_params['tb'])
	pn11_params['L'] = int(pn11_params['tb'] / pn11_params['ts'])
	pn11_params['Npn11'] = pn11_params['Nb'] * pn11_params['L']

	pn11 = generate_pn11_from_params(pn11_params)


	chirp = baseband_chirp_from_params(chirp_params)

	delayed_chirp = np.roll(chirp, 4096)
	delayed_pn11 = np.roll(pn11, 4096)

	plt.plot(chirp, '+-')
	plt.plot(delayed_chirp, '+-')

	chirp_mfilter = MatchedFilterDetector(chirp)
	out = chirp_mfilter.calculate_raw(delayed_chirp)
	# plt.plot(out, '+-')

	plt.figure()
	plt.plot(pn11, '+-')
	plt.plot(delayed_pn11, '+-')

	pn11_mfilter = MatchedFilterDetector(pn11)
	out_pn11 = pn11_mfilter.calculate_raw(delayed_pn11)
	# plt.plot(out_pn11, '+-')

	IPython.embed()


	# B = 2 * np.pi * 0.5
	# Ns = 2048

	# N = int(Ns * 10)

	# # Chirp
	# cx = np.zeros(N, dtype=np.complex64)
	# cx[0:Ns] = baseband_chirp(B, Ns)
	# cf = np.fft.fft(cx)

	# # box
	# bx = np.zeros(N, dtype=float)
	# bx[0:Ns] = np.ones(Ns, dtype=float)
	# bf = np.fft.fft(bx)

	# # Delay signals.
	# cx_d = np.roll(np.copy(cx), 1000)
	# bx_d = np.roll(np.copy(bx), 1000)

	# # Create detectors.
	# chirp_mfilter = MatchedFilterDetector(cx)
	# box_mfilter = MatchedFilterDetector(bx)

	# # Run delayed signals through detectors.
	# craw = chirp_mfilter.calculate_raw(cx_d)
	# braw = box_mfilter.calculate_raw(bx_d)

	# # Plot results.
	# plt.plot(np.abs(craw), label='matched filter output, chirp')
	# plt.plot(np.abs(braw), label='matched filter output, box')
	# plt.legend()

	# IPython.embed()