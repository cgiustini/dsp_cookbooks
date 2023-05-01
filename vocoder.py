import numpy as np
from scipy.io import wavfile
import  scipy.signal as signal
import matplotlib.pyplot as plt
import IPython

def plot_response(w, h, title):
    "Utility function to plot response functions"
    fig = plt.gcf()
    ax = fig.add_subplot(111)
    ax.plot(w, 20*np.log10(np.abs(h)))
    ax.set_ylim(-40, 5)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)

# def get_taps(f, )


# fs, x = wavfile.read(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\aaahhhh.wav")
fs, x = wavfile.read(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\silence.wav")

x = x[:, 0]
# if __name__ == '__main__':

nperseg = 2048

f, t, Sxx = signal.spectrogram(x, fs, nperseg=nperseg)
plt.pcolormesh(t, f, 10 * np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# IPython.embed()

num_filters = 10
filter_size_f_idx = int(np.floor(len(f) / num_filters))

max_f_idx = int(num_filters * filter_size_f_idx)

f = f[0:max_f_idx]
Sxx = Sxx[0:max_f_idx, :]

# plt.pcolormesh(t, f, 10 * np.log10(Sxx))
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()


f_idx = np.reshape(np.arange(len(f)), (num_filters, filter_size_f_idx))

voder_f = []
voder_Sxx = np.zeros((num_filters, len(t)))

for i in np.arange(num_filters):

    this_f_idx = f_idx[i, :]

    voder_f.append(np.mean(f[this_f_idx]))

    voder_Sxx[i, :] = np.sum(Sxx[this_f_idx, :], 0)

    
# plt.pcolormesh(t, np.arange(num_filters), 10 * np.log10(voder_Sxx))

# band = [6000, 8000]  # Desired stop band, Hz
# trans_width = 200    # Width of transition from pass to stop, Hz
# numtaps = 175        # Size of the FIR filter.
# edges = [0, band[0] - trans_width, band[0], band[1],
#          band[1] + trans_width, 0.5*fs]
# taps = signal.remez(numtaps, edges, [1, 0, 1], fs=fs)
# w, h = signal.freqz(taps, [1], worN=2000, fs=fs)
# # plot_response(w, h, "Band-stop Filter")
# # plt.show()

# for i in [10]:
plt.figure()

y = np.zeros(len(t) * nperseg)

# for i in np.arange(len(t)):
for i in [10]:
    all_taps = np.array([], dtype=float)
    print(i, len(t))
    for j in np.arange(len(voder_f)):
    # for j in [3]:
        freq = voder_f[j]
        band = [freq-100, freq+100]  # Desired stop band, Hz
        trans_width = 200    # Width of transition from pass to stop, Hz
        numtaps = 800        # Size of the FIR filter.
        edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
        # print(edges)
        taps = signal.remez(numtaps, edges, [0, 1, 0], fs=fs) * (voder_Sxx[j, i])

        if all_taps.shape == (0,):
            all_taps = taps
        else:
            all_taps = all_taps + taps

    if i is not len(t)-1:
        mean = 0
        std = 1 
        num_samples = nperseg * 10
        samples = np.random.normal(mean, std, size=num_samples)
        this_y = signal.lfilter(all_taps, [1], samples)
        this_y = this_y[len(this_y)-nperseg:len(this_y)]
        y[(i*nperseg):(i+1)*nperseg] = this_y



    w, h = signal.freqz(all_taps, [1], worN=2000, fs=fs)
    plot_response(w, h, "Band-stop Filter")
        

# _, _, Syy = signal.spectrogram(y, fs, nperseg=nperseg)
# Syy = Syy[0:max_f_idx, :]

# plt.plot(f, 10 * np.log10(Sxx[:, 10]))
# plt.plot(f, 10 * np.log10(Syy[:, 10]))
# plt.show()


# y = np.int16(y)
# wavfile.write(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\vocoder.wav", fs, y)

# plt.show()

IPython.embed()