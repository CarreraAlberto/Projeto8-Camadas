from scipy import signal as sg
import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy import signal as window
import matplotlib.pyplot as plt
from scipy.fftpack import fft 

def filtro(y, samplerate, cutoff_hz):
  # https://scipy.github.io/old-wiki/pages/Cookbook/FIRFilter.html
  nyq_rate = samplerate/2
  width = 5.0/nyq_rate
  ripple_db = 60.0 #dB
  N , beta = sg.kaiserord(ripple_db, width)
  taps = sg.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
  yFiltrado = sg.lfilter(taps, 1.0, y)
  return yFiltrado

def LPF(signal, cutoff_hz, fs):
  #####################
  # Filtro
  #####################
  # https://scipy.github.io/old-wiki/pages/Cookbook/FIRFilter.html
  nyq_rate = fs/2
  width = 5.0/nyq_rate
  ripple_db = 60.0 #dB
  N , beta = sg.kaiserord(ripple_db, width)
  taps = sg.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
  return( sg.lfilter(taps, 1.0, signal))

def calcFFT(signal, fs):
    # https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    N  = len(signal)
    W = window.hamming(N)
    T  = 1/fs
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    yf = fft(signal*W)
    return(xf, np.abs(yf[0:N//2]))

def plotFFT(signal, fs):
    x,y = calcFFT(signal, fs)
    plt.figure()
    plt.plot(x, np.abs(y))
    plt.title('Fourier')


# Começo do programa principal
fs = 44100

print("Lendo o áudio")
sound, sampletime = sf.read('sinalModulado.wav')
# audioClear = sound[:,0]
print("Áudio lido")
print()

print("Fazendo FFT do áudio")
plotFFT(sound, fs)
print("FFT feita")
print()

# Tocando o sinal
print("Tocando o sinal")   
sd.play(sound, fs)
sd.wait()
print("Sinal tocado")
print()

# Demodulando o sinal
print("Demodulando o sinal")
t = np.linspace(0, sampletime, len(sound))
portadora = np.sin(2*np.pi*14000*t)
demodulado = sound * portadora
print("Sinal demodulado")
print()

# Filtrando o sinal
print("Filtrando o sinal")
filtrado = filtro(demodulado, fs, 4000)
print("Sinal filtrado")
print()

# Fazendo FFT do sinal
print("Fazendo FFT do sinal")
plotFFT(filtrado, fs)
print("FFT feita")
print()

# Tocar o sinal
print("Tocando o sinal")
sd.play(filtrado, fs)
sd.wait()
print("Sinal tocado")
print()



plt.show()