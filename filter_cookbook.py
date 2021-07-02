import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import IPython

# b, a = signal.ellip(13, 0.009, 80, 0.05, output='ba')
# sos = signal.ellip(13, 0.009, 80, 0.05, output='sos')


N = 4096
fs = 1e6
nyquist = fs / 2
n = np.arange(0, N)
fc = 10e3
wc = 2 * np.pi * fc
x = np.cos(n / fs * wc)

# b = signal.firwin(8, fc  * 0.1)
cutoff = fc * 0.2
cutoff_norm =  cutoff / nyquist
b, a = signal.butter(3, cutoff_norm, btype='low')
y = signal.lfilter(b, a, x)
w, h = signal.freqz(b, a, worN=N, whole=True)




# y_sos = signal.sosfilt(sos, x)
# y_tf = signal.lfilter(b, [], x)

plt.figure(1)
plt.plot(x, 'b+-', label='input')
plt.plot(y, 'ro-', label='TF')
# plt.plot(y_sos, 'k', label='SOS')
plt.legend(loc='best')


sig_fft = np.fft.fft(x)
filtered_fft = np.fft.fft(y)
factor = np.max(abs(sig_fft))

freqs = (fs * 0.5 / np.pi) * w

plt.figure(2)
plt.plot(freqs, 20 * np.log10(abs(sig_fft)), 'b-')
plt.plot(freqs, 20 * np.log10(abs(filtered_fft)), 'r-')
plt.plot(freqs, 20 * np.log10(abs(h)), 'g-', label="freq response", )
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
plt.minorticks_on()

plt.show()

IPython.embed()
