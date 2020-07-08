import numpy as np
import matplotlib.pyplot as plt
import IPython


def baseband_chirp_phase(B, N):

	n = np.arange(N)

	# phase = (B / (2.0 * ((N/2.0) - 1))) * (np.power(n, 2)/2.0 - ((N/2.0) - 1)*n)
	phase = (B/(2*(N-1))) * np.power(n, 2) - (B/2) * n

	return phase

def baseband_chirp(B, N):
	return np.exp(1j * baseband_chirp_phase(B, N))


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

	B = 2 * np.pi * 0.5
	Ns = 2048

	N = int(Ns * 10)

	# Chirp
	cx = np.zeros(N, dtype=np.complex64)
	cx[0:Ns] = baseband_chirp(B, Ns)
	cf = np.fft.fft(cx)

	# box
	bx = np.zeros(N, dtype=float)
	bx[0:Ns] = np.ones(Ns, dtype=float)
	bf = np.fft.fft(bx)

	# Delay signals.
	cx_d = np.roll(np.copy(cx), 1000)
	bx_d = np.roll(np.copy(bx), 1000)

	# Create detectors.
	chirp_mfilter = MatchedFilterDetector(cx)
	box_mfilter = MatchedFilterDetector(bx)

	# Run delayed signals through detectors.
	craw = chirp_mfilter.calculate_raw(cx_d)
	braw = box_mfilter.calculate_raw(bx_d)

	# Plot results.
	plt.plot(np.abs(craw), label='matched filter output, chirp')
	plt.plot(np.abs(braw), label='matched filter output, box')
	plt.legend()

	IPython.embed()