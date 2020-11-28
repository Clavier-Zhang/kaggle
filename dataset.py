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
train_df, valid_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)
test_df = pd.read_csv(TEST_CSV)
# print(train_df.shape, valid_df.shape, test_df.shape)
# print(train_df.head())
# print(valid_df.head())
# print(test_df.head())
#################################

class CnnDataset(Dataset):

    def __init__(self, transforms, X=None, Y=None):
        self.transforms=transforms
        
        self.X = X
        self.Y = Y

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


df = pd.read_csv(csv_path)

    X = list(df['id'])
    y = list(df['category'])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=11)

    y_train, y_valid = LABEL_ENCODER.transform(y_train), LABEL_ENCODER.transform(y_valid)


    train_dataset = CnnDataset(transforms_train, X_train, y_train)
    test_dataset = CnnDataset(transforms_test, X_valid, y_valid)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader, test_dataloader

train_dataloader, test_dataloader = create_cnn_iterator("input/train.csv")


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
