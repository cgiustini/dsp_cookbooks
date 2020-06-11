import numpy as np
import matplotlib.pyplot as plt
import IPython


def baseband_chirp_phase(B, N):

	n = np.arange(N)

	phase = (B / (2.0 * ((N/2.0) - 1))) * (np.power(n, 2)/2.0 - ((N/2.0) - 1)*n)

	return phase



if __name__ == '__main__':

	B = 2 * np.pi * 0.5
	N = 2048
	phase = baseband_chirp_phase(B, N)

	N = int(N * 2)

	# Chirp
	cx = np.zeros(N, dtype=np.complex64)
	cx[0:int(N/2)] = np.exp(1j * phase)
	cf = np.fft.fft(cx)

	# box
	bx = np.zeros(N, dtype=float)
	bx[0:int(N/2)] = np.ones(int(N/2), dtype=float)
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
	cx_d = np.roll(np.copy(cx), int(N/4))
	bx_d = np.roll(np.copy(bx), int(N/4))



	# Perform matched filtering. (currently, doesn't work)
	cf_d = np.fft.fft(np.flip(cx_d))
	bf_d = np.fft.fft(np.flip(bx_d))

	cf_o = np.zeros(N, dtype=np.complex128)
	bf_o = np.zeros(N, dtype=np.complex128)
	for i in range(N):
		cf_o[i] = cf[i] * cf_d[i]
		bf_o[i] = bf[i] * bf_d[i]

	# cx_o = np.fft.ifft(cf_o)
	# bx_o = np.fft.ifft(bf_o)

	# plt.figure()
	# plt.plot(np.fft.fftshift(np.abs(cf_o)), '+-', label='chirp')
	# plt.plot(np.fft.fftshift(np.abs(bf_o)), '+-', label='box')

	IPython.embed()

	# pass