from scipy import signal as sg
import sounddevice as sd
import numpy as np
import soundfile as sf

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

# Começo do programa principal
fs = 44100

print("Lendo o áudio")
sound, sampletime = sf.read('audio.wav')
print("Áudio lido")
print()

print("Filtrando o áudio")
soundFiltrado = LPF(sound, 4000, fs)
print("Áudio filtrado")

print("Tocando o áudio filtrado")
sd.play(soundFiltrado, fs)
sd.wait()
print("Fim da reprodução")



