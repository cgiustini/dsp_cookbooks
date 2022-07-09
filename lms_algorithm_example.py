from numpy import cos, sin, pi, absolute, arange
import numpy as np
from scipy.signal import kaiserord, lfilter, firwin, freqz
import scipy.linalg
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
import IPython

from scipy.io import wavfile

# samplerate1, data1 = wavfile.read(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\going_to_opera.wav")
# samplerate2, data2 = wavfile.read(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\sexy_ass_bitch.wav")

# Ndata = min(data1.shape[0], data2.shape[0])

# data1 = data1[0:Ndata, :]
# data2 = data2[0:Ndata, :]

# data_out = data1 + data2 * 0.2
# data_out = np.int16(data_out)
# wavfile.write(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\combined_audio.wav", samplerate1, data_out)



def run_lms(x, d, N, Ntaps):

    # Initialize matrix representing the LMS filter estimate.
    w_lms =  np.zeros((Ntaps, N))

    # Initialize arrays to keep track of the
    # LMS filter output, the LMS error, and true
    # error between filter output and true filter output (uncorrupted by noise).
    y = np.zeros(N)
    e = np.zeros(N)
    etrue = np.zeros(N)

    # Run LMS algorithm.
    for i in np.arange(Ntaps, N-1):
        u = x[np.arange(i, i-Ntaps, -1)]
        y[i] = np.dot(w_lms[:, i], u)
        e[i] = d[i] - y[i]
        etrue[i] = xf[i] - y[i]
        mu = 0.008
        w_lms[:, i+1] = w_lms[:, i] + mu * u * e[i]

    return w_lms, y, e



if __name__ == '__main__':

    # Number of tests.
    # N=1000

    # Kaiser filter.
    sample_rate = 100.0
    nyq_rate = sample_rate / 2.0
    width = 5.0/nyq_rate
    ripple_db = 60.0
    Ntaps, beta = kaiserord(ripple_db, width)
    cutoff_hz = 10.0
    w = firwin(Ntaps, cutoff_hz/nyq_rate, window=('kaiser', beta))

    # Delay filter.
    # Ntaps = 16
    # w = np.zeros(Ntaps, dtype=float)
    # kDelay = 1
    # w[kDelay] = 1.0

    # Generate the data.
    # x = np.random.normal(0.0, 1.0, N) 
    # n = np.random.normal(0.0, 1.0, N) * 0.1
    # xf = lfilter(w, 1.0, x)
    # d = xf + n

    samplerate1, data1 = wavfile.read(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\going_to_opera.wav")
    samplerate2, data2 = wavfile.read(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\sexy_ass_bitch.wav")
    Ndata = min(data1.shape[0], data2.shape[0])
    data1 = data1[0:Ndata, :]
    data2 = data2[0:Ndata, :]

    # data_out = data1 + data2 * 0.2
    # data_out = np.int16(data_out)
    # wavfile.write(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\combined_audio.wav", samplerate1, data_out)

    x = np.float64(data1[:, 0]) / np.power(2, 15) # signal that gets echo'ed
    n = np.float64(data2[:, 0]) / np.power(2, 15)# signal that will be preserved after echo cancellation
    xf = lfilter(w, 1.0, x) # echo
    d = xf + n # signal to remove echo from

    N=len(d)

    w_lms, y, e = run_lms(x, d, N, Ntaps)

    figure(1)
    plot(xf, 'o-')
    plot(y, 'o-')
    plot(e, 'ro-')

    figure(2)
    plot(w_lms[:, np.arange(0, N, 10)], 'o-')
    plot(w, 'ro-')
    show()

    data_out_with_echo = np.int16(d * np.power(2, 15))
    wavfile.write(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\audio_with_echo.wav", samplerate1, data_out_with_echo)
    data_out_without_echo = np.int16(e * np.power(2, 15))
    wavfile.write(r"C:\Users\Carlo Giustini\Desktop\dsp_cookbooks\samples\audio_without_echo.wav", samplerate1, data_out_without_echo)

    import IPython
    IPython.embed()