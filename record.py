import sounddevice as sd
import scipy.io.wavfile as wav
import os

fs=16000
duration = 2  # seconds
name = input("enter your name:")
newpath = r'b:/recordings/{}'.format(name)
if not os.path.exists(newpath):
    os.makedirs(newpath)
    dataset = 1
    while(dataset == 1):
        label = input("enter label of dataset:")
        newpath1 = r'b:/recordings/{}/{}'.format(name, label)
        if not os.path.exists(newpath1):
            os.makedirs(newpath1)
            for i in range(20):
                myrecording = sd.rec(duration * fs, samplerate=fs, channels=1,dtype='float32')
                print ("Recording \'{}\'{}:".format(label,i+1))
                sd.wait()
                os.chdir('b:/recordings/{}/{}'.format(name, label))
                wav.write("{}_{}-{}.wav" .format(name,label,str(i+1)), fs, myrecording)
        else:
            print("filename already exists!!!")

        dataset = int(input("Enter 1 to continue or  any key to stop:"))
else:
    print("filename already exists!!!!")