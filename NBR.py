
from preprocess import *
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
#from keras.utils.vis_utils import plot_model

#used only for model picture generation
#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


# Second dimension of the feature is dim2
feature_dim_2 = 11

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# # Feature dimension
feature_dim_1 = 20
channel = 1

# Predicts one sample
def predict(filepath, model):
    sample = wav2mfcc(filepath)

    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    label = get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]
    if(label == '0not_detected'):
        label = "Sorry,Can't hear your voice!!"
    return label

#Our Predefined model
model = load_model('my_model.h5')

#plot_model(model, to_file='model1.png', show_shapes=True, show_layer_names=True) #generates model image
#print(model.summary()) #generates model summary

keyboard=1
while(keyboard==1):
    #predcition

    import pyaudio
    import wave
    from pydub import AudioSegment
     
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = "file.wav"
     
    audio = pyaudio.PyAudio()
 
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print ("recording...")
    frames = []
     
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")
     
     
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
     
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    sound = AudioSegment.from_wav("file.wav")
    sound = sound.set_channels(1)
    sound.export("monofile.wav", format="wav")
    print("We predicted you saying : ")
    print(predict('monofile.wav', model=model))
    keyboard = int(input("Enter 1 to try again"))
