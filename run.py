from deep_speech.mozilla_deep_speech.mozillaDeepSpeech import get_model, get_text_from_audio, get_audio_file
from deep_speech.utils.constants import modelPath, scorerPath, lm_alpha, lm_beta, beam_width, audio_file

if __name__=="__main__":
    model = get_model(modelPath, scorerPath, lm_alpha, lm_beta, beam_width)
    audio_file = get_audio_file(audio_file)
    print("The text from audio is: " , get_text_from_audio(model, audio_file))