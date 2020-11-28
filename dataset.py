import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import train_test_split
from tools import LABEL_ENCODER
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGES_DIR = 'input/images/'
TRAIN_CSV = 'input/train.csv'
TEST_CSV = 'input/test.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 11

#################################
# Split data
df = pd.read_csv(TRAIN_CSV)
df['category'] = LABEL_ENCODER.transform(df['category'])
TRAIN_DF, VALID_DF = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
TRAIN_DF.reset_index(drop=True, inplace=True)
VALID_DF.reset_index(drop=True, inplace=True)
TEST_DF = pd.read_csv(TEST_CSV)
print(TRAIN_DF.shape, VALID_DF.shape, TEST_DF.shape)
print(TRAIN_DF.head())
print(VALID_DF.head())
print(pd.merge(TRAIN_DF, VALID_DF, how ='inner', on =['id']) )
# print(TEST_DF.head())
#################################


TRANSFORM_NORMAL = A.Compose([
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])


class CnnDataset(Dataset):

    def __init__(self, transforms, df):
        self.transforms=transforms
        self.X = list(df['id'])
        self.Y = list(df['category'])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):

        image_path = IMAGES_DIR + str(self.X[idx]) + '.jpg'
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        return image, self.Y[idx]

def CREATE_CNN_TRAIN_ITERATOR(transforms, batch_size):
    cnn_train_dataset = CnnDataset(transforms, TRAIN_DF)
    cnn_train_iterator = DataLoader(cnn_train_dataset, batch_size=batch_size, shuffle=True)
    return cnn_train_iterator

def CREATE_CNN_VALID_ITERATOR(batch_size):
    cnn_valid_dataset = CnnDataset(TRANSFORM_NORMAL, VALID_DF)
    cnn_valid_iterator = DataLoader(cnn_valid_dataset, batch_size=batch_size, shuffle=True)
    return cnn_valid_iterator




#################################

class TestDataset(Dataset):

    def __init__(self, df):
        self.transforms = A.Compose([
            A.Normalize(p=1.0),
            ToTensorV2(p=1.0),
        ])
        self.IDs = list(df['id'])
  

    def __len__(self):
        return len(self.IDs)
    
    def __getitem__(self, idx):

        image_path = IMAGES_DIR + str(self.IDs[idx]) + '.jpg'
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        return self.IDs[idx], image
