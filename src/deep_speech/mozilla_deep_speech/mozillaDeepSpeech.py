import deepspeech
import wave
import numpy as np

def get_model(modelPath, scorerPath, lm_alpha, lm_beta, beam_width):
    model = deepspeech.Model(modelPath)
    model.enableExternalScorer(scorerPath)
    model.setScorerAlphaBeta(lm_alpha,lm_beta)
    model.setBeamWidth(beam_width)
    return model

def get_audio_file(audio_file):
    wave_file = wave.open(audio_file,'r')
    frames = wave_file.getnframes()
    buffer = wave_file.readframes(frames)
    data16 = np.frombuffer(buffer,dtype=np.int16)
    return data16

def get_text_from_audio(model, audio_file):
    text = model.stt(audio_file)
    return text

