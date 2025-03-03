import pandas as pd
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Encode categories into numbers
LABEL_ENCODER = LabelEncoder()
LABEL_ENCODER.fit(pd.read_csv("input/train.csv")['category'])


# X encoder
GENDER_ENCODER = LabelEncoder()
GENDER_ENCODER.fit(pd.read_csv("input/train.csv")['gender'])

COLOR_ENCODER = LabelEncoder()
COLOR_ENCODER.fit(pd.read_csv("input/train.csv")['baseColour'])

SEASON_ENCODER = LabelEncoder()
SEASON_ENCODER.fit(pd.read_csv("input/train.csv")['season'])

USAGE_ENCODER = LabelEncoder()
USAGE_ENCODER.fit(pd.read_csv("input/train.csv")['usage'])


transforms_train = A.Compose([
    # A.Flip(),
    A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),

    # Pixels
    A.OneOf([
        A.IAAEmboss(p=1.0),
        A.IAASharpen(p=1.0),
        A.Blur(p=1.0),
    ], p=0.5),

    # Affine
    # A.OneOf([
    #     A.ElasticTransform(p=1.0),
    #     A.IAAPiecewiseAffine(p=1.0)
    # ], p=0.5),

    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

transforms_test = A.Compose([
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])
