import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.enc_layer1 = nn.Linear(28, 16)
        self.enc_layer2 = nn.ReLU()
        self.enc_layer3 = nn.Linear(16, 10)
        self.enc_layer4 = nn.ReLU()
        self.enc_layer5 = nn.Linear(10, 4)

        self.dec_layer1 = nn.Linear(4, 10)
        self.dec_layer2 = nn.ReLU()
        self.dec_layer3 = nn.Linear(10, 16)
        self.dec_layer4 = nn.ReLU()
        self.dec_layer5 = nn.Linear(16, 28)

    def forward(self, x):
        x = self.enc_layer1(x)
        x = self.enc_layer2(x)
        x = self.enc_layer3(x)
        x = self.enc_layer4(x)
        x = self.enc_layer5(x)

        x = self.dec_layer1(x)
        x = self.dec_layer2(x)
        x = self.dec_layer3(x)
        x = self.dec_layer4(x)
        x = self.dec_layer5(x)

        return x


dataframe = pd.read_csv('creditcard.csv')
dataframe.drop(['Time', 'Amount'], axis=1, inplace=True)
train_data, test_data = train_test_split(dataframe, test_size=0.2, random_state=27)

test_data = pd.concat([test_data, train_data[train_data.Class == 1]])
normal_test_count = len(test_data[test_data.Class == 0])
fraud_test_count = len(test_data[test_data.Class == 1])
test_label = test_data['Class']
test_data.drop(['Class'], axis=1, inplace=True)
test_data = test_data.values
test_label = test_label.values
print('normal_test_count:', normal_test_count)
print('fraud_test_count:', fraud_test_count)

train_data = train_data[train_data.Class == 0]  # deleting fraudulent data from train data
train_data.drop(['Class'], axis=1, inplace=True)
train_data = train_data.values

# training the model
auto_encoder = AutoEncoder().double()
epochs = 25
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(epochs):
    loss_hist = np.array([])
    for data in train_loader:
        output = auto_encoder(data)
        loss = loss_func(output, data)
        loss_hist = np.append(loss_hist, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_loss = np.mean(loss_hist)
    print('epoch:', epoch + 1, '; loss:', mean_loss)

# Evaluation
mse_from_test = []
auto_encoder.eval()  # go into eval, stops training
with torch.no_grad():
    for data in test_loader:
        output = auto_encoder(data)
        loss = loss_func(output, data).data.item()
        mse_from_test.append(loss)

MSE_label_df = pd.DataFrame(mse_from_test)
MSE_label_df = MSE_label_df.rename(columns={0: 'mse'})
MSE_label_df['label'] = test_label
print(MSE_label_df.head())

# Results
permissible_error = 7.3
output_prediction = [1 if e > permissible_error else 0 for e in MSE_label_df.mse.values]
confusion_matrix = confusion_matrix(MSE_label_df.label, output_prediction)

MSE_normal = 0
MSE_fraud = 0

normal_indices = [i for i, x in enumerate(test_label) if x == 0]
fraud_indices = [i for i, x in enumerate(test_label) if x == 1]

for i in normal_indices:
    MSE_normal += mse_from_test[i]
for i in fraud_indices:
    MSE_fraud += mse_from_test[i]

MSE_normal = MSE_normal / normal_test_count
MSE_fraud = MSE_fraud / fraud_test_count

TP = confusion_matrix[1][1]
FP = confusion_matrix[1][0]
TN = confusion_matrix[0][0]
FN = confusion_matrix[0][1]
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 / ((1 / precision) + (1 / recall))

print(confusion_matrix)
print("precision =", precision)
print("recall =", recall)
print("F1_score =", f1_score)
print("MSE for normal =", MSE_normal)
print("MSE for fraud =", MSE_fraud)
