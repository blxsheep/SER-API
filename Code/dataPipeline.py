


import pandas as pd
from sqlalchemy import create_engine
import os 
import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
import pandas as pd
import torch.nn.functional as F
import torchvision
import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torchsummary import summary
from tqdm  import tqdm
import torch.optim as optim
import datetime
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio










res_dir  = 'Code/classification_res'
db_url = 'db_url'
table_name = 'table_name'
data_path = '/content/drive/MyDrive/migrated_SER/SER- Project/src_as_wav'
test_df= pd.read_csv('/content/drive/MyDrive/migrated_SER/SER- Project/label_tables/rest_df.csv')
model_path = 'saved_models/audioutil_x_efficient_b0.ipynb'
input_dir = 'Code/input_audio'




class AudioUtil():
  # ----------------------------
  # Load an audio file. Return the signal as a tensor and the sample rate
  # ----------------------------
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)
   # ----------------------------
  # Convert the given audio to the desired number of channels
  # ----------------------------
  @staticmethod
  def rechannel(aud, new_channel):
    sig, sr = aud

    if (sig.shape[0] == new_channel):
      # Nothing to do
      return aud

    if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
      resig = sig[:1, :]
    else:
      # Convert from mono to stereo by duplicating the first channel
      resig = torch.cat([sig, sig])

    return ((resig, sr))
     # ----------------------------
  # Since Resample applies to a single channel, we resample one channel at a time
  # ----------------------------
  @staticmethod
  def resample(aud, newsr):
    sig, sr = aud

    if (sr == newsr):
      # Nothing to do
      return aud

    num_channels = sig.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    if (num_channels > 1):
      # Resample the second channel and merge both channels
      retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
      resig = torch.cat([resig, retwo])

    return ((resig, newsr))
     # ----------------------------
  # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
  # ----------------------------
  @staticmethod
  def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)
      
    return (sig, sr)
      # ----------------------------
  # Shifts the signal to the left or right by some percent. Values at the endx
  # are 'wrapped around' to the start of the transformed signal.
  # ----------------------------
  @staticmethod
  def time_shift(aud, shift_limit):
    sig,sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)
   # ----------------------------
  # Generate a Spectrogram
  # ----------------------------

  @staticmethod
  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)
  # ----------------------------
  # Augment the Spectrogram by masking out some sections of it in both the frequency
  # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
  # overfitting and to help the model generalise better. The masked sections are
  # replaced with the mean value.
  # ----------------------------
  @staticmethod
  def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec
  


# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
  def __init__(self, df, data_path,augment = True):
    self.df = df
    self.data_path = str(data_path)
    self.duration = 4000
    self.sr = 44100
    self.channel = 1
    self.shift_pct = 0.4
    self.augment = augment
            
  # ----------------------------
  # Number of items in dataset
  # ----------------------------
  def __len__(self):
    return len(self.df)    
    
  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
  def __getitem__(self, idx):
    # Absolute file path of the audio file - concatenate the audio directory with
    # the relative path
    audio_file = self.data_path +'/' +self.df.loc[idx, 'relative_path']
    # Get the Class ID


    aud = AudioUtil.open(audio_file)
    # Some sounds have a higher sample rate, or fewer channels compared to the
    # majority. So make all sounds have the same number of channels and same 
    # sample rate. Unless the sample rate is the same, the pad_trunc will still
    # result in arrays of different lengths, even though the sound duration is
    # the same.
    reaud = AudioUtil.resample(aud, self.sr)
    rechan = AudioUtil.rechannel(reaud, self.channel)

    dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
    if self.augment :
      shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
    else : 
      shift_aud = dur_aud
    sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    if self.augment :
      aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    else :
      aug_sgram = sgram
    transform = torchvision.transforms.Resize((224,224))
    aug_sgram = transform(aug_sgram)
    fake_rgb = torch.stack([aug_sgram.squeeze(), aug_sgram.squeeze(), aug_sgram.squeeze()],dim =0)




    return fake_rgb 



def load_df_to_postgres(df, table_name, db_url):
    """
    Load a Pandas DataFrame to a PostgreSQL database table using SQLAlchemy.

    Parameters:
    df (pandas.DataFrame): The DataFrame to load to the database.
    table_name (str): The name of the table to create in the database.
    db_url (str): The SQLAlchemy connection string for the database.
                  Example: 'postgresql://user:password@localhost:5432/mydatabase'

    Returns:
    None
    """
    # Create a SQLAlchemy engine object for the database
    engine = create_engine(db_url)

    # Create the table in the database using the DataFrame schema
    df.head(0).to_sql(table_name, engine, if_exists='replace', index=False)

    # Load the DataFrame data to the database table in batches
    batch_size = 1000
    num_batches = int(df.shape[0] / batch_size) + 1
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        df_batch = df.iloc[start:end]
        df_batch.to_sql(table_name, engine, if_exists='append', index=False)

    print(f"Loaded {df.shape[0]} rows to {table_name} table in the database.")



def classify(data_path_dir) :
    input_audio_list = os.listdir(data_path_dir)
    df = pd.DataFrame({'relative_path': input_audio_list})
    #df = df[:10]
    batch_size= 1
    val_ds = SoundDS(df,data_path_dir, augment =False)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False )
    valid_data_size = batch_size * len(val_dl)

    checkpoint = torch.load(model_path)
    model = checkpoint['model']
    optimizer_state_dict = checkpoint['optimizer']

    # Set the model to evaluation mode
    model.eval()

    prediction = []
    for inputs in val_dl:
    # Pass the input to the model
        outputs = model(inputs)

        # Apply softmax activation
        probs = torch.softmax(outputs, dim=1)

        # Get the predicted class
        _, predicted = torch.max(probs.data, 1)

        # Print the predicted class
        print(predicted.item())
        prediction.append(predicted.item())
    emotion_dict = {0: 'Neutral' , 1: 'Angry'   , 2 : 'Happy', 3 : 'Sad' , 4: 'Frustrated'}
    res_df = df.copy()
    res_df['output'] =prediction
    res_df['emotion'] = res_df['output'].map(emotion_dict)
    # save res_df
    datetimenow = datetime.datetime.now()

# Create a filename with the current date and time
    filename = "emotion_res_" + str(datetimenow)
    res_df.to_csv(os.join(res_dir,filename))
    emotion_cnt = res_df['emotion'].value_counts()
    res = emotion_cnt.to_json()
    return res


### incomplete
def splitter(input_dir):
  # Define the original file path and filename
    for filename in os.listdir(input_dir):
        original_file_path = os.path.join(input_dir,filename)
        filename_base = original_file_path.replace('.wav','')

        # Load the audio file
        waveform, sample_rate = torchaudio.load(original_file_path)

        # Determine the length of the audio in seconds
        audio_length = waveform.size(1) / sample_rate

        # Split the audio into segments of length <= 20 seconds
        segment_length = 20 * sample_rate
        segments = []
        for i in range(0, waveform.size(1), segment_length):
            segment = waveform[:, i:i+segment_length]
            segments.append(segment)
            
        # If there is any remaining audio, add it as a final segment
        if waveform.size(1) % segment_length != 0:
            segment = waveform[:, i+segment_length:]
            segments.append(segment)

        # Delete the original audio file
        os.remove(original_file_path)

        # Save each segment as a separate audio file with the original filename
        for i, segment in enumerate(segments):
            segment_length = segment.size(1) / sample_rate
            filename = f"{filename_base}_{i*20}_{(i+1)*20-1}_({segment_length:.2f}s).wav"
            filepath = os.path.join(os.path.dirname(original_file_path), filename)
            torchaudio.save(filepath, segment, sample_rate)

        # Confirm that all segmented audio files were saved successfully
        num_segments = len(segments)
        num_files = len(os.listdir(os.path.dirname(original_file_path)))
        assert num_segments == num_files, f"Expected {num_segments} files but found {num_files} instead"
    
def upload_res():
    for file in os.listdir(res_dir): 
        if '.csv' in file : 
            df = pd.read_csv(os.path.join(res_dir,file))
        load_df_to_postgres(df , table_name , db_url)
    print('upload_all_file')
    return


