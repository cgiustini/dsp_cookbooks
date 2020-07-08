import numpy as np
import matplotlib.pyplot as plt
import IPython

from fmcw import MatchedFilterDetector, ClassicFMCWDetector, baseband_chirp

if __name__ == '__main__':

	B = 2 * np.pi * 0.1
	Ns = 2048

	N = Ns

	# Chirp
	cx = np.zeros(N, dtype=np.complex64)
	cx[0:Ns] = baseband_chirp(B, Ns)
	cf = np.fft.fft(cx)

	# box
	bx = np.zeros(N, dtype=float)
	bx[0:Ns] = np.ones(Ns, dtype=float)
	bf = np.fft.fft(bx)

	# Delay signals.
	cx_d = np.roll(np.copy(cx), 1024)
	bx_d = np.roll(np.copy(bx), 1024)
		

	# Create detectors.
	chirp_mfilter = MatchedFilterDetector(cx)
	box_mfilter = MatchedFilterDetector(bx)

	chirp_classic = ClassicFMCWDetector(cx)

	# Run delayed signals through detectors.
	# craw = chirp_mfilter.calculate_raw(cx_d)
	# braw = box_mfilter.calculate_raw(bx_d)

	Nfs = 1000
	fs = np.linspace(-0.2, 0.2, Nfs)
	# Nfs = 1
	# fs = [0]

	doppler_mfilter = np.zeros((Nfs, N), dtype=np.complex128)
	doppler_classic = np.zeros((Nfs, N), dtype=np.complex128)

	for i, f in enumerate(fs):

		phasor =  np.exp(-1j * 2 * np.pi * f * np.arange(N))
		cx_d_dshift = np.multiply(cx_d, phasor)
		craw = chirp_mfilter.calculate_raw(cx_d_dshift)
		doppler_mfilter[i, :] = craw
		craw = chirp_classic.calculate_raw(cx_d_dshift)
		doppler_classic[i, :] = craw

	plt.figure()
	plt.imshow(20 * np.log10(np.abs(doppler_mfilter)))
	plt.title('Matched Filter FMCW')
	plt.figure()
	plt.imshow(20 * np.log10(np.abs(doppler_classic)))
	plt.title('Classic FMCW')
	# plt.imshow(np.abs(doppler))
		


		# return np.exp(1j * 2 * np.pi * f * np.arange(N))

	# Plot results.
	# plt.plot(np.abs(craw), label='matched filter output, chirp')
	# plt.plot(np.abs(braw), label='matched filter output, box')
	# plt.legend()

	IPython.embed()