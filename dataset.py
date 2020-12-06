import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
import jovian
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
from collections import Counter
from sklearn.model_selection import train_test_split

from tools import LABEL_ENCODER, GENDER_ENCODER, SEASON_ENCODER, COLOR_ENCODER, USAGE_ENCODER

# pca
from sklearn import decomposition
from sklearn import datasets


#tokenization
tok = spacy.load('en')
def tokenize (text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]

def encode_sentence(text, vocab2index, N=15):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length



def preprocess_dataset(df):

    df['description'] = df['noisyTextDescription']
    df['description_length'] = df['description'].apply(lambda x: len(x.split()))

    #count number of occurences of each word
    counts = Counter()
    for index, row in df.iterrows():
        counts.update(tokenize(row['description']))
    
    #deleting infrequent words
    # print("num_words before:",len(counts.keys()))
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]
    # print("num_words after:",len(counts.keys()))

    #creating vocabulary
    vocab2index = {"":0, "UNK":1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    df['encoded'] = df['description'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))



    df['gender'] = GENDER_ENCODER.transform(df['gender'])
    df['season'] = SEASON_ENCODER.transform(df['season'])
    df['baseColour'] = COLOR_ENCODER.transform(df['baseColour'])
    df['usage'] = USAGE_ENCODER.transform(df['usage'])

    return df, words, counts


def preprocess_test_dataset(df, counts):


    df['description'] = df['noisyTextDescription']
    df['description_length'] = df['description'].apply(lambda x: len(x.split()))

    #creating vocabulary
    vocab2index = {"":0, "UNK":1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    df['encoded'] = df['description'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))

    # encode X
    df['gender'] = GENDER_ENCODER.transform(df['gender'])
    df['season'] = SEASON_ENCODER.transform(df['season'])
    df['baseColour'] = COLOR_ENCODER.transform(df['baseColour'])
    df['usage'] = USAGE_ENCODER.transform(df['usage'])

    return df




IMAGES_DIR = 'input/images/'
TRAIN_CSV = 'input/train.csv'
TEST_CSV = 'input/test.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 11

#################################
# Split data
df = pd.read_csv(TRAIN_CSV)
df, words, counts = preprocess_dataset(df)


df['category'] = LABEL_ENCODER.transform(df['category'])
train_df, valid_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)



test_df = pd.read_csv(TEST_CSV)
test_df = preprocess_test_dataset(test_df, counts)

# print(train_df.shape, valid_df.shape, test_df.shape)
print(train_df.head())
# print(valid_df.head())
# print(pd.merge(train_df, valid_df, how ='inner', on =['id']) )
# print(test_df.head())
#################################







def read_image(dir_path, id, transforms=None):
    # use cv2 to read imgaes
    image_path = dir_path + str(id) + '.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if transforms:
        transformed = transforms(image=image)
        image = transformed['image']
    return image




class CnnDataset(Dataset):

    def __init__(self, transforms, df):
        self.transforms=transforms
        self.IDs = list(df['id'])
        self.descriptions = list(df['encoded'])
        self.labels = list(df['category'])


        self.genders = list(df['gender'])
        self.seasons = list(df['season'])
        self.colors = list(df['baseColour'])
        self.usages = list(df['usage'])

    def __len__(self):
        return len(self.IDs)
    
    def __getitem__(self, idx):

        # read image
        image = read_image(dir_path=IMAGES_DIR, id=self.IDs[idx], transforms=self.transforms)


        # read description
        description = torch.from_numpy(self.descriptions[idx][0].astype(np.int32))
        l = self.descriptions[idx][1]

        # read label
        label = self.labels[idx]
        
        # read X
        features = torch.Tensor([self.genders[idx], self.seasons[idx], self.colors[idx], self.usages[idx]])


        return image, (description, l), features, label

def CREATE_CNN_TRAIN_ITERATOR(transforms, batch_size):
    cnn_train_dataset = CnnDataset(transforms, train_df)
    cnn_train_iterator = DataLoader(cnn_train_dataset, batch_size=batch_size, shuffle=True)
    return cnn_train_iterator

def CREATE_CNN_VALID_ITERATOR(batch_size):
    transform = A.Compose([
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ])
    cnn_valid_dataset = CnnDataset(transform, valid_df)
    cnn_valid_iterator = DataLoader(cnn_valid_dataset, batch_size=batch_size, shuffle=True)
    return cnn_valid_iterator




class TestDataset(Dataset):

    def __init__(self, df):
        self.transforms= A.Compose([
            A.Normalize(p=1.0),
            ToTensorV2(p=1.0),
        ])
        self.IDs = list(df['id'])
        self.descriptions = list(df['encoded'])
        self.IDs = list(df['id'])


        self.genders = list(df['gender'])
        self.seasons = list(df['season'])
        self.colors = list(df['baseColour'])
        self.usages = list(df['usage'])

    def __len__(self):
        return len(self.IDs)
    
    def __getitem__(self, idx):

        # read image
        image = read_image(IMAGES_DIR, self.IDs[idx])
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        description = torch.from_numpy(self.descriptions[idx][0].astype(np.int32))
        l = self.descriptions[idx][1]

        features = torch.Tensor([self.genders[idx], self.seasons[idx], self.colors[idx], self.usages[idx]])

        return image, (description, l), self.IDs[idx], features

TEST_DATASET = TestDataset(test_df)

