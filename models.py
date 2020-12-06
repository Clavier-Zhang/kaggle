import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


OUTPUT_DIM = 27
class CNN(nn.Module):
    
    def __init__(self, num_classes=OUTPUT_DIM):
        super().__init__()
        self.backbone = torchvision.models.wide_resnet101_2(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.logit = nn.Linear(in_features, num_classes)


        
    def forward(self, x):
        batch_size, C, H, W = x.shape
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        
        x = F.dropout(x, 0.25, self.training)

        x = self.logit(x)

        return x


VOCAB_SIZE = 7000
EMBEDDING_DIM = 100
HIDDEN_DIM = 50

class RNN(nn.Module) :
    
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM) :
        super().__init__()
        self.dropout = nn.Dropout(0.3)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 27)
        
    def forward(self, x, s):


        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out


class NN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size) :
        super().__init__()
        hidden_size = 27
        self.fc1 = nn.Linear(input_size, hidden_size)

        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size+4, hidden_size)

        self.dropout = nn.Dropout(0.75)

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        temp = x
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)

        x = self.fc6(x)
        x = self.relu(x)

        x = torch.cat((temp, x), dim=1)

        x = self.fc7(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x.squeeze()

