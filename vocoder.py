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
    # ax.set_ylim(-40, 5)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)

# def get_taps(f, )

# f = [130, ]

def get_formants(fft_x, num_formats=10, idx_tol=200):

    max_idx = np.flip(np.argsort(fft_x))

    formant_idx = np.array([max_idx[0]])
    for m in max_idx[1:-1]:

        if len(formant_idx) >= num_formats:
            break

        if all(abs(formant_idx-m) > idx_tol):
            formant_idx = np.append(formant_idx, m)

    formant_amplitudes = fft_x[formant_idx]

    return formant_idx, formant_amplitudes

def generate_synthetic_signal_from_formants(formant_idx, formant_amplitudes, L):
    y = np.zeros(L, dtype=float)
    for i in np.arange(len(formant_idx)):
        y = y + np.sin(2 * np.pi * formant_idx[i] / (N*2) * np.arange(L)) * formant_amplitudes[i]
    return y

def lpf(y, cutoff, fs):       
    band = [cutoff]  # Desired stop band, Hz
    trans_width = 50   # Width of transition from pass to stop, Hz
    numtaps = 4096       # Size of the FIR filter.
    edges = [0, band[0], band[0] + trans_width, 0.5*fs]
    taps = signal.remez(numtaps, edges, [1, 0], fs=fs)
    y = signal.lfilter(taps, [1], y)
    return y

def generate_vocoder_filter(formant_idx, formant_amplitudes, max_formant_idx, fs, L):

    all_taps = np.array([], dtype=float)

    # for i in np.arange(len(formant_idx)):
    for i in [0, 3, 5, 7]:

        print(i)

        # freq = float(formant_idx[i] * fs / max_formant_idx)
        # band = [freq - 100, freq + 100]
        # trans_width = 200
        # numtaps = 375
        # edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]

        # taps = signal.remez(numtaps, edges, [0, 1, 0], fs=fs) * formant_amplitudes[i]

        freq = float(formant_idx[i] * fs / max_formant_idx)
        band = [freq-10, freq+10]  # Desired stop band, Hz
        trans_width = 10    # Width of transition from pass to stop, Hz
        numtaps = 8192       # Size of the FIR filter.
        edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
        # print(edges)
        taps = signal.remez(numtaps, edges, [0, 1, 0], fs=fs) * formant_amplitudes[i]

        if all_taps.shape == (0,):
            all_taps = taps
        else:
            all_taps = all_taps + taps
    
    return all_taps

fs, x = wavfile.read(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\aaahhhh.wav")
# fs, x = wavfile.read(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\silence.wav")

x = x[:, 0]


N = int(np.floor(len(x)/2))
fft_x = np.fft.fft(x)
fft_x = abs(fft_x[0:N])

freq = np.arange(N) / N *  (fs/2)

formant_idx, formant_amplitudes = get_formants(fft_x, num_formats=10, idx_tol=200)

# plt.plot(freq, abs(fft_x))

# for m in formant_idx:
#     plt.axvline(m, color='red')


L = N * 2
y = generate_synthetic_signal_from_formants(formant_idx, formant_amplitudes, L)

# all_taps = generate_vocoder_filter(formant_idx, formant_amplitudes, L, fs, L)
# samples = np.random.normal(0, 1, size=L)
# y = signal.lfilter(all_taps, [1], lpf(samples, 2000, fs))
fft_y = np.fft.fft(y)
fft_y = abs(fft_y[0:N])

# w, h = signal.freqz(all_taps, [1], worN=2000, fs=fs)
# # plt.plot(freq, 20 * np.log10(fft_x))
# plot_response(w, h, "Band-stop Filter")
plt.plot(freq, 20 * np.log10(fft_x))
plt.plot(freq, 20 * np.log10(fft_y))


y = y / np.max(abs(y)) * np.max(abs(x))

y = np.int16(y)
wavfile.write(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\vocoder.wav", fs, y)

# f = np.arange(0, len(x)) / len(x) * fs

# kernel_size = 100
# kernel = np.ones(kernel_size) / kernel_size
# fft_x_smooth = np.convolve(fft_x, kernel, mode='same')


# plt.plot(f, abs(fft_x))

# plt.plot(f, 10 * np.log10(fft_x))
# plt.plot(f, 10 * np.log10(fft_x_smooth))
# plt.ion()
# plt.show()

# x = x[:, 0]
# # if __name__ == '__main__':

# nperseg = 2048

# f, t, Sxx = signal.spectrogram(x, fs, nperseg=nperseg)
# plt.pcolormesh(t, f, 10 * np.log10(Sxx))
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

# IPython.embed()

# num_filters = 10
# filter_size_f_idx = int(np.floor(len(f) / num_filters))

# max_f_idx = int(num_filters * filter_size_f_idx)

# f = f[0:max_f_idx]
# Sxx = Sxx[0:max_f_idx, :]

# # plt.pcolormesh(t, f, 10 * np.log10(Sxx))
# # plt.ylabel('Frequency [Hz]')
# # plt.xlabel('Time [sec]')
# # plt.show()


# f_idx = np.reshape(np.arange(len(f)), (num_filters, filter_size_f_idx))

# voder_f = []
# voder_Sxx = np.zeros((num_filters, len(t)))

# for i in np.arange(num_filters):

#     this_f_idx = f_idx[i, :]

#     voder_f.append(np.mean(f[this_f_idx]))

#     voder_Sxx[i, :] = np.sum(Sxx[this_f_idx, :], 0)

    
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
# plt.figure()

# y = np.zeros(len(t) * nperseg)

# # for i in np.arange(len(t)):
# for i in [10]:
#     all_taps = np.array([], dtype=float)
#     print(i, len(t))
#     for j in np.arange(len(voder_f)):
#     # for j in [3]:


#         if all_taps.shape == (0,):
#             all_taps = taps
#         else:
#             all_taps = all_taps + taps

#     if i is not len(t)-1:
#         mean = 0
#         std = 1 
#         num_samples = nperseg * 10
#         samples = np.random.normal(mean, std, size=num_samples)
#         this_y = signal.lfilter(all_taps, [1], samples)
#         this_y = this_y[len(this_y)-nperseg:len(this_y)]
#         y[(i*nperseg):(i+1)*nperseg] = this_y



#     w, h = signal.freqz(all_taps, [1], worN=2000, fs=fs)
#     plot_response(w, h, "Band-stop Filter")
        

# _, _, Syy = signal.spectrogram(y, fs, nperseg=nperseg)
# Syy = Syy[0:max_f_idx, :]

# plt.plot(f, 10 * np.log10(Sxx[:, 10]))
# plt.plot(f, 10 * np.log10(Syy[:, 10]))
# plt.show()


# y = np.int16(y)
# wavfile.write(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\vocoder.wav", fs, y)

# plt.show()

IPython.embed()