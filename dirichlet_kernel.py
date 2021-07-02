import numpy as np
import IPython
import matplotlib.pyplot as plt
import IPython

if __name__ == '__main__':

	Ns = np.arange(1, 4096)
	wc = 0.01


	d = []
	for N in Ns:
		d.append(np.sum(np.cos(wc * np.arange(0, N))))

	dq = np.sin(wc * Ns / 2) / np.sin(wc / 2) * np.cos(wc * (Ns-1) / 2)

	plt.plot(Ns, d, '+-')
	plt.plot(Ns, dq, 'o')

	IPython.embed()
