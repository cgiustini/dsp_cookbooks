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

	# Time domain plot.
	plt.figure()
	plt.plot(np.real(cx), '+-', label='chirp')
	plt.plot(bx, '+-', label='box')

	# Frequency domain plot.
	plt.figure()
	plt.plot(np.fft.fftshift(np.abs(cf)), '+-', label='chirp')
	plt.plot(np.fft.fftshift(np.abs(bf)), '+-', label='box')

	# Delay signals.
	cx_d = np.roll(np.copy(cx), 1000)
	bx_d = np.roll(np.copy(bx), 1000)

	# Perform matched filtering. (currently, doesn't work)
	cf_d = np.fft.fft(np.flip(np.conj(cx_d)))
	bf_d = np.fft.fft(np.flip(np.conj(bx_d)))

	# cf_o = np.zeros(N, dtype=np.complex128)
	# bf_o = np.zeros(N, dtype=np.complex128)
	# for i in range(N):
	# 	cf_o[i] = cf[i] * cf_d[i]
	# 	bf_o[i] = bf[i] * bf_d[i]

	cf_o = np.multiply(cf, cf_d)
	bf_o = np.multiply(bf, bf_d)
	cx_o = np.fft.ifft(cf_o)
	bx_o = np.fft.ifft(bf_o)


	# cf_o = np.convolve(cx_d, np.flip(np.conj(cx)), 'same')

	# plt.figure()
	# plt.plot(np.fft.fftshift(np.abs(cf_o)), '+-', label='chirp')
	# plt.plot(np.fft.fftshift(np.abs(bf_o)), '+-', label='box')

	IPython.embed()

	# pass