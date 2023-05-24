import numpy as np
from scipy.io import wavfile
import  scipy.signal as signal
import matplotlib.pyplot as plt
import IPython

from moviepy.editor import VideoClip
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from moviepy.video.io.bindings import mplfig_to_npimage


def make_movie(figs):
    # duration of the video
    duration = 2

    fps = 10
    duration = len(figs) / fps
    
    # method to get frames
    def make_frame(t):
        
        i = int(t * 10)

        print(i)

        # returning numpy image
        return mplfig_to_npimage(figs[i])
    
    # creating animation
    animation = VideoClip(make_frame, duration = duration)
    
    # displaying animation with auto play and looping
    animation.ipython_display(fps = fps, loop = False, autoplay = False)

def plot_response(w, h, title):
    "Utility function to plot response functions"
    fig = plt.gcf()
    ax = fig.add_subplot(111)
    ax.plot(w, 20*np.log10(np.abs(h)))
    ax.set_xscale('log')
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


def generate_vocoder_filter_2(formant_idx, formant_amplitudes, max_formant_idx, fs, L):

    edges = [0]
    desired = [0]

    # for i in np.arange(len(formant_idx)):
    for i in [0, 3,5 ,7]:

        freq = float(formant_idx[i] * fs / max_formant_idx)
        band = [freq-50, freq+50]  # Desired stop band, Hz
        trans_width = 100    # Width of transition from pass to stop, Hz
        
        first_edge = band[0] - trans_width 
        first_edge = first_edge if first_edge >=0 else 0
        edges.append(first_edge)
        edges.append(band[0])
        edges.append(band[1])
        edges.append(band[1] + trans_width)

        desired.append(formant_amplitudes[i] / np.max(formant_amplitudes))
        desired.append(0)

    edges.append(0.5*fs)
    


    # IPython.embed()    

    numtaps = 8192
    taps = signal.remez(numtaps, edges, desired, fs=fs) 
    
    return taps

def generate_vocoder_filter_3(x, band_freqs, band_amplitudes, max_formant_idx, fs):

    y = np.zeros(x.shape)
    h = np.zeros(max_formant_idx)

    for i in np.arange(band_freqs.shape[0]):
    # for i in [0]:

        # [freq - freq * 0.3, freq + freq * 0.3]
        freq = band_freqs[i, :]
        # print(freq)
        b, a = signal.butter(3, freq, 'bandpass', fs=fs, analog=False)
        # b, a = signal.butter(5, 3000, 'low', fs=fs, analog=False)
        b = b * band_amplitudes[i]

        w, this_h = signal.freqz(b,a, worN=max_formant_idx, fs=fs)
        # plot_response(w, h, "Band-stop Filter")
        h = this_h + h

        this_y = signal.lfilter(b, a, x)
        y = y + this_y

    # plt.figure()
    
    # plot_response(w, h, "Band-stop Filter")
    # plt.plot(formant_freq, 20 * np.log10(formant_amplitudes), 'o-')

    return (y, h)

def generate_vocoder_filter_4(band_freqs, band_amplitudes, max_formant_idx, fs):

    h = np.zeros(max_formant_idx)
    Nord = 3

    a_list = []
    b_list = []

    for i in np.arange(band_freqs.shape[0]):
    # for i in [0]:

        # [freq - freq * 0.3, freq + freq * 0.3]
        freq = band_freqs[i, :]
        # print(freq)
        b, a = signal.butter(Nord, freq, 'bandpass', fs=fs, analog=False)
        # b, a = signal.butter(5, 3000, 'low', fs=fs, analog=False)
        b = b * band_amplitudes[i]

        a_list.append(a)
        b_list.append(b)

        w, this_h = signal.freqz(b, a, worN=max_formant_idx, fs=fs)
        # plot_response(w, h, "Band-stop Filter")
        h = this_h + h

        # this_y = signal.lfilter(b, a, x)
        # y = y + this_y

    # plt.figure()
    
    # plot_response(w, h, "Band-stop Filter")
    # plt.plot(formant_freq, 20 * np.log10(formant_amplitudes), 'o-')

    return (b_list, a_list, h)

def apply_vocoder_filter(x, b_list, a_list):

    y = np.zeros(x.shape)

    for i in np.arange(len(b_list)):
        b = b_list[i]
        a = a_list[i]

        this_y = signal.lfilter(b, a, x)
        y = y + this_y

    return y

# b, a = signal.butter(3, [300, 500], 'bandpass', fs=fs, analog=False)
# w, h = signal.freqz(b,a, worN=2000, fs=fs)
# plot_response(w, h, "Band-stop Filter")

def freq_to_fft_idx(freq, N, fs):
    fft_idx = np.round(freq / (fs/2) * N)
    return fft_idx.astype(int)

def fft_idx_to_freq(fft_idx, N, fs):
    freq = fft_idx / N *  (fs/2)
    return freq



# x = x[0:44000]

def run_vocoder_on_chunk(x, w):

    N = int(np.floor(len(x)/2))

    fft_x = np.fft.fft(x)
    fft_x = abs(fft_x[0:N])
    fft_w = np.fft.fft(w)
    fft_w = abs(fft_w[0:N])

    plot_freq = np.arange(N) / N *  (fs/2)

    start_freq = 100
    stop_freq = 10000

    # filter_bank_freq_edges = np.arange(np.log10(start_freq(), 2500, step=400, dtype=float)

    # filter_bank_freq_edges = np.arange(0, 2500, step=400, dtype=float)

    # filter_bank_freq_bands = []
    # filter_bank_freq_centers = []

    # for i in np.arange(len(filter_bank_freq_edges)-1):
    #     band = [filter_bank_freq_edges[i], filter_bank_freq_edges[i+1]]
    #     filter_bank_freq_bands.append(np.array(band))
    #     filter_bank_freq_centers.append(np.mean(band))

    num_bands = 16
    bw_factor = 0.3
    filter_bank_freq_centers = np.logspace(np.log10(start_freq), np.log10(stop_freq), num_bands)
    # print(filter_bank_freq_centers)
    filter_bank_freq_bands = []
    for i in np.arange(len(filter_bank_freq_centers)):
        freq = filter_bank_freq_centers[i]
        band = [freq - freq * bw_factor, freq + bw_factor]
        filter_bank_freq_bands.append(np.array(band))

    filter_bank_freq_bands = np.array(filter_bank_freq_bands)
    # filter_bank_freq_centers = np.array(filter_bank_freq_centers)

    filter_bank_idx_bands = freq_to_fft_idx(filter_bank_freq_bands, N, fs)
    filter_bank_idx_centers = freq_to_fft_idx(filter_bank_freq_centers, N, fs)

    filter_bank_freq_pwrs = []

    for b in filter_bank_idx_bands:
        idx_to_sum = np.arange(b[0], b[1])
        mean_pwr = np.sum(fft_x[idx_to_sum]) / len(idx_to_sum)
        filter_bank_freq_pwrs.append(mean_pwr)

    filter_bank_freq_pwrs = np.array(filter_bank_freq_pwrs) / np.max(filter_bank_freq_pwrs)

    # (y, h) = generate_vocoder_filter_3(w, filter_bank_freq_bands, filter_bank_freq_pwrs, N, fs)
    (b_list, a_list, h) = generate_vocoder_filter_4(filter_bank_freq_bands, filter_bank_freq_pwrs, N, fs)
    y = apply_vocoder_filter(w, b_list, a_list)

    # y = y / np.sqrt(np.mean(np.power(y,2,dtype=float))) * np.sqrt(np.mean(np.power(x,2,dtype=float))) * 100
    y =  np.int16(y / np.max(abs(y)) * np.max(abs(x)))

    fft_y = np.fft.fft(y)
    fft_y = abs(fft_y[0:N])
    fft_w = np.fft.fft(w)
    fft_w = abs(fft_w[0:N])
    
    fig = plt.figure()
    plt.plot(plot_freq, 20 * np.log10(fft_x/ np.max(fft_x)))
    plt.plot(plot_freq, 20 * np.log10(abs(h)  / np.max(abs(h))))
    # plt.plot(filter_bank_freq_centers, 20 * np.log10(filter_bank_freq_pwrs), 'o-')
    ax = plt.gca()
    ax.set_xlim([50, 20000])
    ax.set_xscale('log')
    
    
    # plt.figure()
    # plt.plot(plot_freq, 20 * np.log10(fft_w/ np.max(fft_w)))
    # plt.plot(plot_freq, 20 * np.log10(abs(h)  / np.max(abs(h))))
    # plt.plot(plot_freq, 20 * np.log10(fft_y/ np.max(fft_y)))
    # ax = plt.gca()
    # ax.set_xscale('log')

    # IPython.embed()


    return (y, fig)

# audio_file_x = r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\aaahhhh_2.wav"
audio_file_x = r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\zoo.wav"
# audio_file_x = r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\keyboard_recording_2.wav"
# audio_file_x = r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\ah_oh.wav"

audio_file_w = r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\keyboard_recording.wav"

fs, x = wavfile.read(audio_file_x)
fs, w = wavfile.read(audio_file_w)
# w = np.random.randn(len(x))

# fs, x = wavfile.read(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\ssshhh.wav")
# fs, x = wavfile.read(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\ah_sh.wav")
# fs, x = wavfile.read(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\ah_oh.wav")
# fs, x = wavfile.read(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\zoo.wav")
# fs, w = wavfile.read(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\keyboard_recording.wav")
# fs, x = wavfile.read(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\silence.wav")


x = x[:, 0]
w = w[:, 0]


# chunk_size = int(fs/2)
chunk_size = int(fs/20)

num_chunks = int(np.floor(len(x) / chunk_size))

final_size = chunk_size * num_chunks
time_idx = np.arange(final_size)
time_idx_chunks = np.reshape(time_idx, (num_chunks, chunk_size))
x_chunks = np.reshape(x[0:final_size], (num_chunks, chunk_size))
w_chunks = np.reshape(w[0:final_size], (num_chunks, chunk_size))


y_chunks = np.zeros(x_chunks.shape, dtype=np.int16)

figs = []
for i in np.arange(x_chunks.shape[0]):
    print(i, x.shape[0])
    (y_chunks[i, :], this_fig) = run_vocoder_on_chunk(x_chunks[i, :], w_chunks[i, :])
    figs.append(this_fig)
    plt.close(this_fig)

    # if i==3:
        # IPython.embed()

    plt.figure(1000)
    plt.plot(time_idx_chunks[i, :], y_chunks[i, :])

    plt.figure(1001)
    plt.plot(time_idx_chunks[i, :], x_chunks[i, :])
        # plt.plot(x_chunks[i, :] / np.max(abs(x_chunks[i, :])))
        # plt.plot(w_chunks[i, :] / np.max(abs(w_chunks[i, :])))
        # plt.plot(y_chunks[i, :] / np.max(abs(y_chunks[i, :])))

y = np.reshape(y_chunks, final_size)

# make_movie(figs)

# video_clip = VideoFileClip(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\__temp__.mp4")
# audio_clip = AudioFileClip(audio_file)
# final_clip = video_clip.set_audio(audio_clip)
# final_clip.write_videofile("final.mp4")


# y = run_vocoder_on_chunk(x, w)

# plt.plot(freq, 20 * np.log10(fft_x))
# plt.plot(freq, 20 * np.log10(fft_w))
# plt.plot(freq, 20 * np.log10(fft_y))

# wavfile.write(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\vocoder_ah_oh.wav", fs, y)
# wavfile.write(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\vocoder_ah2.wav", fs, y)
wavfile.write(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\vocoder_ah2.wav", fs, y)


# formant_idx, formant_amplitudes = get_formants(fft_x, num_formats=10, idx_tol=200)

# # plt.plot(freq, abs(fft_x))

# # for m in formant_idx:
# #     plt.axvline(m, color='red')


# L = N * 2
# y = generate_synthetic_signal_from_formants(formant_idx, formant_amplitudes, L)

# all_taps = generate_vocoder_filter_2(formant_idx, formant_amplitudes, L, fs, L)

# all_taps = generate_vocoder_filter_2(formant_idx, formant_amplitudes, L, fs, L)

# samples = np.random.normal(0, 1, size=L)
# y = generate_vocoder_filter_3(w, formant_idx, formant_amplitudes, L, fs, L)
# fft_y = np.fft.fft(y)
# fft_y = abs(fft_y[0:N])

# w, h = signal.freqz(all_taps, [1], worN=2000, fs=fs)
# plot_response(w, h, "Band-stop Filter")




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