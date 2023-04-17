
import pandas as pd
from sqlalchemy import create_engine
import os 
import  random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
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
import io
import boto3
from io import BytesIO




#sql config
db_url = 'db_url'
table_name = 'table_name'

#s3 config
bucket = "thaiser-file-storage"
region = "ap-southeast-2"
access_key = "AKIA6OROF7LYVCXDAAEO"
secret_key = "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"
client = boto3.client('s3', aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key, region_name=region)
res_dir  = 's3://thaiser-file-storage/res/'#res/
data_path_dir = 's3://thaiser-file-storage/input_audio/' #input_audio/
model_path = 's3://thaiser-file-storage/saved_models/eff_b0_extend10linear.pt'#saved_models/
hist_data_path_dir ='s3://thaiser-file-storage/history_input/'#history_input/
buffer_input_dir ='s3://thaiser-file-storage/buffer_input/'
 

def s3_listdir(s3_uri):
    # Parse the S3 URI into bucket name and prefix
    s3_parts = s3_uri.split('/', 3)
    bucket_name = s3_parts[2]
    prefix = '' if len(s3_parts) == 3 else s3_parts[3]

    # Initialize the S3 client
    s3 = client

    # List the objects in the bucket with the specified prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # Extract the object keys from the response
    object_keys = [obj['Key'] for obj in response.get('Contents', [])]

    # Return the object keys
    return object_keys
def parse_s3_uri(uri):
    parsed_uri = boto3.s3.parse_s3_uri(uri)
    return parsed_uri['bucket'], parsed_uri['key']
def load_model(s3_uri):
    # Parse the S3 URI
    s3 = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    bucket_name, key = s3_uri.replace("s3://", "").split("/", 1)
    
    # Load the model file from S3 to memory
    obj = s3.Object(bucket_name, key)
    file_stream = BytesIO()
    obj.download_fileobj(file_stream)
    file_stream.seek(0)
    
    # Load the model from the file stream using torch.load()
    model = torch.load(file_stream, map_location=torch.device('cpu'))
    
    return model

# def s3_listdir(s3_uri):
#     # Parse the S3 URI into bucket name and prefix
#     s3_parts = s3_uri.split('/', 3)
#     bucket_name = s3_parts[2]
#     prefix = '' if len(s3_parts) == 3 else s3_parts[3]

#     # Initialize the S3 client
#     s3 = client

#     # List the objects in the bucket with the specified prefix
#     response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

#     # Extract the object keys from the response
#     object_keys = [obj['Key'] for obj in response.get('Contents', [])]

#     # Return the object keys
#     return object_keys

def move_to_destination_folder(src, dest):
    """
    Moves a file from a source S3 bucket and key to a destination S3 bucket and key.
    
    Parameters:
    src (str): The S3 URI of the source file to move.
    dest (str): The S3 URI of the destination directory to move the file to.
    """
    # Parse source S3 bucket and key
    src_parts = src.replace('s3://', '').split('/')
    src_bucket = src_parts[0]
    src_key = '/'.join(src_parts[1:])

    # Parse destination S3 bucket and key
    dest_parts = dest.replace('s3://', '').split('/')
    dest_bucket = dest_parts[0]
    dest_key = '/'.join(dest_parts[1:])

    # Create S3 client
    s3_client = client
    print('pass check')

    # Move file to destination folder
    s3_client.copy_object(
        Bucket=dest_bucket,
        CopySource={'Bucket': src_bucket, 'Key': src_key},
        Key=f"{dest_key}{src_parts[-1]}"
    )

    # Delete file from source folder

    print('src_bucket : ' , src_bucket)
    print('src_key :', src_key)
    s3_client.delete_object(
        Bucket=src_bucket,
        Key=src_key
    )
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
# class SoundDS(Dataset):
#   def __init__(self, df, data_path,augment = True):
#     self.df = df
#     self.data_path = str(data_path)
#     self.duration = 4000
#     self.sr = 44100
#     self.channel = 1
#     self.shift_pct = 0.4
#     self.augment = augment
            
#   # ----------------------------
#   # Number of items in dataset
#   # ----------------------------
#   def __len__(self):
#     return len(self.df)    
    
#   # ----------------------------
#   # Get i'th item in dataset
#   # ----------------------------
#   def __getitem__(self, idx):
#     # Absolute file path of the audio file - concatenate the audio directory with
#     # the relative path
#     audio_file_key = self.df.loc[idx, 'relative_path']
#     audio_object = self.s3.get_object(Bucket=self.bucket_name, Key=audio_file_key)
#     audio_data = audio_object['Body'].read()
#     # Get the Class ID
#     with io.BytesIO(audio_data) as f:
#       aud = AudioUtil.open(f)
#     # Some sounds have a higher sample rate, or fewer channels compared to the
#     # majority. So make all sounds have the same number of channels and same 
#     # sample rate. Unless the sample rate is the same, the pad_trunc will still
#     # result in arrays of different lengths, even though the sound duration is
#     # the same.
#     reaud = AudioUtil.resample(aud, self.sr)
#     rechan = AudioUtil.rechannel(reaud, self.channel)

#     dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
#     if self.augment :
#       shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
#     else : 
#       shift_aud = dur_aud
#     sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
#     if self.augment :
#       aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
#     else :
#       aug_sgram = sgram
#     transform = torchvision.transforms.Resize((224,224))
#     aug_sgram = transform(aug_sgram)
#     fake_rgb = torch.stack([aug_sgram.squeeze(), aug_sgram.squeeze(), aug_sgram.squeeze()],dim =0)




#     return fake_rgb 


# class SoundDS(Dataset):
#   def __init__(self, df, data_path_dir,augment=True):
#     bucket = "thaiser-file-storage"
#     region = "ap-southeast-2"
#     access_key = "acc_key"
#     secret_key = "secret_key"
#     client = boto3.client('s3', aws_access_key_id=access_key,
#                           aws_secret_access_key=secret_key, region_name=region)
#     self.df = df
#     self.s3_client = client
#     self.bucket_name = bucket
#     self.data_path_dir = data_path_dir
#     self.duration = 4000
#     self.sr = 44100
#     self.channel = 1
#     self.shift_pct = 0.4
#     self.augment = augment
            
#   def __len__(self):
#     return len(self.df)    
    
#   def __getitem__(self, idx):
#     file_name = self.df.loc[idx, 'relative_path']
#     audio_file_key = f"{self.data_path_dir}/{file_name}"
#     audio_object = self.s3_client.get_object(Bucket=self.bucket_name, Key=audio_file_key)
#     print(audio_file_key)
#     audio_data = audio_object['Body'].read()
#     with io.BytesIO(audio_data) as f:
#       aud = AudioUtil.open(f)
#     reaud = AudioUtil.resample(aud, self.sr)
#     rechan = AudioUtil.rechannel(reaud, self.channel)

#     dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
#     if self.augment:
#       shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
#     else: 
#       shift_aud = dur_aud
#     sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
#     if self.augment:
#       aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
#     else:
#       aug_sgram = sgram
#     transform = torchvision.transforms.Resize((224,224))
#     aug_sgram = transform(aug_sgram)
#     fake_rgb = torch.stack([aug_sgram.squeeze(), aug_sgram.squeeze(), aug_sgram.squeeze()], dim=0)

#     return fake_rgb
class SoundDS(Dataset):
    def __init__(self, df, data_path_dir, augment=True):
        self.df = df
        bucket = "thaiser-file-storage"
        region = "ap-southeast-2"
        access_key = "AKIA6OROF7LYVCXDAAEO"
        secret_key = "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"
        client = boto3.client('s3', aws_access_key_id=access_key,
                              aws_secret_access_key=secret_key, region_name=region)
        self.s3_client = client
        self.bucket_name = "thaiser-file-storage"
        self.data_path_dir = data_path_dir
        self.duration = 4000
        self.sr = 44100
        self.channel = 1
        self.shift_pct = 0.4
        self.augment = augment
            
    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        audio_file_key = self.df.loc[idx, 'relative_path']
       
        audio_object = self.s3_client.get_object(Bucket=self.bucket_name, Key=audio_file_key)
        audio_data = audio_object['Body'].read()
        with io.BytesIO(audio_data) as f:
            aud = AudioUtil.open(f)
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        if self.augment:
            shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        else: 
            shift_aud = dur_aud
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        if self.augment:
            aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        else:
            aug_sgram = sgram
        transform = torchvision.transforms.Resize((224,224))
        aug_sgram = transform(aug_sgram)
        fake_rgb = torch.stack([aug_sgram.squeeze(), aug_sgram.squeeze(), aug_sgram.squeeze()], dim=0)

        return fake_rgb
def post_df_to_s3(res_df, file_name):
  csv_buffer = io.StringIO()
  res_df.to_csv(csv_buffer, index=False)

  # Create a new S3 object with the specified key
  key = 'res/' + file_name
  client.put_object(Body=csv_buffer.getvalue(), Bucket=bucket, Key=key)
  print('Posted csv to s3')
  return


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



def classify(user_id = 1) :
    input_audio_list = s3_listdir(data_path_dir)
    df = pd.DataFrame({'relative_path': input_audio_list})
    print(df.head())
    #df = df[:10]
    batch_size= 1
    val_ds = SoundDS(df,data_path_dir, augment =False)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False )
    valid_data_size = batch_size * len(val_dl)

    checkpoint = load_model(model_path)
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

      # Get the predicted class indices for all inputs in the batch
      predicted_indices = torch.argmax(probs, dim=1)

      # Print the predicted classes for each input in the batch
      for i in range(batch_size):
          print(predicted_indices[i].item())
          prediction.append(predicted_indices[i].item())

    emotion_dict = {0: 'Neutral' , 1: 'Angry'   , 2 : 'Happy', 3 : 'Sad' , 4: 'Frustrated'}
    res_df = df.copy()
    res_df['output'] =prediction
    res_df['emotion'] = res_df['output'].map(emotion_dict)
    res_df['user_id']  = user_id
    # save res_df
    datetimenow = datetime.datetime.now()

# Create a filename with the current date and time
    filename = "emotion_res_" + str(datetimenow) +'.csv'
    post_df_to_s3(res_df, filename)
    emotion_cnt = res_df['emotion'].value_counts()
    res = emotion_cnt.to_json()

    destination_folder = hist_data_path_dir
    for f in s3_listdir(data_path_dir):
      #move file to history after classify
  # Full path to the file to be moved
      file_path = data_path_dir+f.replace('input_audio/','') #remove redundant
    

    # Move the file to the destination folder
      print('src :' ,file_path)
      print('des :',destination_folder)
      move_to_destination_folder(file_path, destination_folder)
    return res


## incomplete
# def splitter(data_path_dir = data_path_dir):
#   # Define the original file path and filename
#     for filename in os.listdir(data_path_dir):
#         original_file_path = os.path.join(data_path_dir,filename)
#         filename_base = original_file_path.replace('.wav','')

#         # Load the audio file
#         waveform, sample_rate = torchaudio.load(original_file_path)

#         # Determine the length of the audio in seconds
#         audio_length = waveform.size(1) / sample_rate

#         # Split the audio into segments of length <= 20 seconds
#         segment_length = 20 * sample_rate
#         segments = []
#         for i in range(0, waveform.size(1), segment_length):
#             segment = waveform[:, i:i+segment_length]
#             segments.append(segment)
            
#         # If there is any remaining audio, add it as a final segment
#         if waveform.size(1) % segment_length != 0:
#             segment = waveform[:, i+segment_length:]
#             segments.append(segment)

#         # Delete the original audio file
#         os.remove(original_file_path)

#         # Save each segment as a separate audio file with the original filename
#         for i, segment in enumerate(segments):
#             segment_length = segment.size(1) / sample_rate
#             filename = f"{filename_base}_{i*20}_{(i+1)*20-1}_({segment_length:.2f}s).wav"
#             filepath = os.path.join(os.path.dirname(original_file_path), filename)
#             torchaudio.save(filepath, segment, sample_rate)

#         # Confirm that all segmented audio files were saved successfully
#         num_segments = len(segments)
#         num_files = len(os.listdir(os.path.dirname(original_file_path)))
#         assert num_segments == num_files, f"Expected {num_segments} files but found {num_files} instead"

# def splitter():
#     # Create an S3 client
#     s3 = client
    
#     # List all objects in the directory
#     prefix ='input_audio/'
#     # Define the original file path and filename for each object
#     for i in s3_listdir(data_path_dir):
#       print(i)
#     for relative_path in s3_listdir(data_path_dir):
#         filename = relative_path.replace('input_audio/','')
#         print('relative_path :', relative_path)
  
#         s3_uri = f"s3://{bucket}/{relative_path}"
#         filename_base = s3_uri.replace('.wav', '')

#         # Load the audio file
#         #waveform, sample_rate = torchaudio.load(s3_uri)
#         audio_object = client.get_object(Bucket=bucket, Key=relative_path)
#         audio_data = audio_object['Body'].read()
#         with io.BytesIO(audio_data) as f:
#             waveform , sample_rate = torchaudio.load(f)

#         # Determine the length of the audio in seconds
#         audio_length = waveform.size(1) / sample_rate

#         # Split the audio into segments of length <= 20 seconds
#         segment_length = 20 * sample_rate
#         segments = []
#         for i in range(0, waveform.size(1), segment_length):
#             segment = waveform[:, i:i+segment_length]
#             segments.append(segment)

#         # If there is any remaining audio, add it as a final segment
#         if waveform.size(1) % segment_length != 0:
#             segment = waveform[:, i+segment_length:]
#             segments.append(segment)

#         # Delete the original audio file
#         s3.delete_object(Bucket=bucket, Key=relative_path)

#         # Save each segment as a separate audio file with the original filename
#         for i, segment in enumerate(segments):
#             segment_length = segment.size(1) / sample_rate
#             filename = f"{filename_base}_{i*20}_{(i+1)*20-1}_({segment_length:.2f}s).wav"
#             s3_segment_uri = f"s3://{bucket}/{prefix}/{filename}".replace('s3://thaiser-file-storage/input_audio//','')
            
#             torchaudio.save(s3_segment_uri, segment, sample_rate)

#         # Confirm that all segmented audio files were saved successfully
#         num_segments = len(segments)
#         response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
#         num_files = len(response['Contents'])
#         assert num_segments == num_files, f"Expected {num_segments} files but found {num_files} instead"
import tempfile

def splitter():
    # Create an S3 client
    s3 = client
    
    # List all objects in the directory
    prefix ='buffer_input/'

    # Define the original file path and filename for each object
    for i in s3_listdir(buffer_input_dir):
      print(i)
    for relative_path in s3_listdir(buffer_input_dir):
        filename = relative_path.replace(prefix,'')
        print('relative_path :', relative_path)
  
        s3_uri = f"s3://{bucket}/{relative_path}"
        filename_base = s3_uri.replace('.wav', '')

        # Load the audio file
        #waveform, sample_rate = torchaudio.load(s3_uri)
        audio_object = client.get_object(Bucket=bucket, Key=relative_path)
        audio_data = audio_object['Body'].read()
        with io.BytesIO(audio_data) as f:
            waveform , sample_rate = torchaudio.load(f)

        # Determine the length of the audio in seconds
        audio_length = waveform.size(1) / sample_rate

        # Split the audio into segments of length <= 20 seconds
        segment_length = 20 * sample_rate
        segments = []
        for i in range(0, waveform.size(1), segment_length):
            segment = waveform[:, i:i+segment_length]
            segments.append(segment)

        # If there is any remaining audio, add it as a final segment
        if waveform.size(1) % segment_length > 1 and waveform.size(1) > segment_length:
            segment = waveform[:, i+segment_length:]
            segments.append(segment)

        # Delete the original audio file
        s3.delete_object(Bucket=bucket, Key=relative_path)

        # Save each segment as a separate audio file with the original filename
        for i, segment in enumerate(segments):
            segment_length = segment.size(1) / sample_rate
            filename = f"{filename_base}_{i*20}_{(i+1)*20-1}_({segment_length:.2f}s).wav".replace('s3://thaiser-file-storage/buffer_input/','')
            des = f"{'input_audio'}/{filename}"
            print('des : ', des)
            with tempfile.NamedTemporaryFile() as tmp_file:
                print(f"Temporary file created: {tmp_file.name}")
                torchaudio.save(tmp_file.name +'.wav', segment, sample_rate)
                s3.upload_file(tmp_file.name+'.wav', bucket,des )

      

# def upload_res():
#     for file in s3_listdir(res_dir): 
#         if '.csv' in file : 
#             df = pd.read_csv(os.path.join(res_dir,file))
#         load_df_to_postgres(df , table_name , db_url)
#     print('upload_all_file')
#     return
def s3_load_csv():
  s3 = client
  res_dir = 'res/'

  # List all CSV files in the S3 bucket
  response = s3.list_objects_v2(Bucket=bucket, Prefix=res_dir)
  csv_files = [f['Key'] for f in response['Contents'] if f['Key'].endswith('.csv')]

  # Read each CSV file and append its data frame to a list
  df_list = []
  for csv_file in csv_files:
      # Load the CSV file from S3
      obj = s3.get_object(Bucket=bucket, Key=csv_file)
      # Read the CSV data into a data frame
      df = pd.read_csv(obj['Body'])
      # Append the data frame to the list
      df_list.append(df)
  return df_list
def upload_res():
  dfs = s3_load_csv()
  print('Loading .',end= '')
  for df in dfs:
    load_df_to_postgres(df, table_name,db_url)
    print('.' ,end =' ')
  print()
  print(' Loaded to postgress')
