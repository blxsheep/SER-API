from tensorflow import keras
import json
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.python.keras.utils import  img_to_array
# from tensorflow.python.keras.utils import  load_img
from tensorflow.keras.utils import load_img, img_to_array

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_model(path='D:\Documents\KMITL\Project1\Code\models\EffNetZero'):
    try:
        model = keras.models.load_model(path)
        return model
    except:
        return 'Load model Fail'


def classify(input_audio_path='D:\Documents\KMITL\Project1\Code\input_audio'):
  model  = load_model()
  for filename in os.listdir(input_audio_path):
    audio_path  = os.path.join(input_audio_path,filename)
    toImage(audio_path)
    

  #toImage(input_audio_path)
  input_list = []
  input_img_path  ='D:\Documents\KMITL\Project1\Code\input_image'
  for filename in os.listdir(input_img_path):
      image_path =  os.path.join(input_img_path,filename)
      img =  prep_for_model(image_path)
      input_list.append(img)
  input_list=  np.array(input_list)
  output_layer =model.predict(input_list)
  output_list = translate_to_Emotion(output_layer)
      
  return output_list




def toImage(audio_path):
    image_folder_path = 'D:\Documents\KMITL\Project1\Code\input_image'
    filename = '\\' + os.path.basename(audio_path)
    image_path = image_folder_path + filename.replace('.wav','.png')
    y, sr = librosa.load(audio_path)  # your file
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), fmax=8000)
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    


def prep_fn(img):  # prepare for NN
    img = img.astype(np.float32) / 255.0  # [0,1]
    # img = (img - 0.5) * 2  # [0 -0.5 , 1 - 0.5] => [-0.5, 0.5] * 2 => [-1,1]
    return img


def prep_for_model(img_path, img_size=(256, 256, 3)):
    img = img_to_array(load_img(img_path, grayscale=False, color_mode='rgb',
                       target_size=img_size, interpolation='bilinear'))
    img = prep_fn(img)
    return img


def translate_to_Emotion(list_of_res):
    emo_list = []
    for output_layer in list_of_res:
        emo_dict = {1: 'Neutral', 2: 'Angry', 3: 'Happy',
                    4: 'Sad', 5: 'Frustrated', 0: 'None'}
        output_layer = output_layer.tolist()
        index = output_layer.index(max(output_layer))
        emo_list.append(emo_dict[index])
    return emo_list
