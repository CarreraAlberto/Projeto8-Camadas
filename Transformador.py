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
    plt.title('Fourier (antes da modulacao)')

def plotFFT1(signal, fs):
    x,y = calcFFT(signal, fs)
    plt.figure()
    plt.plot(x, np.abs(y))
    plt.title('Fourier (depois da modulacao)')


# Começo do programa principal
fs = 44100

print("Lendo o áudio")
sound, sampletime = sf.read('audio.wav')
print("Áudio lido")
print()

print("Extraindo da lista")
audioClear = sound[:,0]
audioClear = audioClear/np.max(np.abs(audioClear))
print("Extraído")
print()

print("Filtrando o áudio")
soundFiltrado = LPF(audioClear, 4000, fs)
print("Áudio filtrado")
print()

# Plot do sinal de audio original normalizado pelo tempo
print("Plotando o sinal de audio original normalizado pelo tempo")
plt.figure()
plt.plot(sound)
plt.title('Sinal de audio original normalizado pelo tempo')
print("Plotado")
print()

# Plot do sinal de audio filtrado normalizado pelo tempo
print("Plotando o sinal de audio filtrado normalizado pelo tempo")
plt.figure()
plt.plot(soundFiltrado)
plt.title('Sinal de audio filtrado normalizado pelo tempo')
print("Plotado")
print()

print("Tocando o áudio filtrado")
sd.play(soundFiltrado, fs)
sd.wait()
print("Fim da reprodução")
print()

# Plotando a FFT
print("Plotando a FFT")
plotFFT(soundFiltrado, fs)
print("FFT plotada")
print()

# Modulando o sinal de audio em AM com portadora de 14kHz
print("Modulando o sinal de audio em AM com portadora de 14kHz")
t = np.linspace(0, sampletime, len(soundFiltrado))
portadora = np.sin(2*np.pi*14000*t)
sinalModulado = soundFiltrado*portadora
print("Sinal modulado")
print()

# Normalize esse sinal: multiplicar o sinal por uma constante (a maior possível), de modo que todos os pontos do sinal permaneçam dentro do intervalo[-1,1]
print("Normalizando o sinal")
sinalModulado = sinalModulado/np.max(np.abs(sinalModulado))
sf.write('sinalModulado.wav', sinalModulado, fs)
print("Sinal normalizado")
print()

# Tocando o sinal modulado
print("Tocando o sinal modulado")
sd.play(sinalModulado, fs)
sd.wait()
print("Fim da reprodução")
print()

# Plot do sinal modulado pelo tempo
print("Plotando o sinal modulado pelo tempo")
plt.figure()
plt.plot(sinalModulado)
plt.title('Sinal modulado pelo tempo')
print("Plotado")
print()

# Plotando a FFT do sinal modulado
print("Plotando a FFT do sinal modulado")
plotFFT1(sinalModulado, fs)
print("FFT plotada")
print()

plt.show()