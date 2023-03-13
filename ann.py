import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import sys
import cv2
sys.path.append('..')

# Create CNN Model
class ANN_Model(nn.Module):
    def __init__(self):
        super(ANN_Model, self).__init__()

        self.fc1 = nn.Linear(36,18)
        self.relu1 = nn.ReLU() # activation
        self.fc2 = nn.Linear(18,9)
        self.relu2 = nn.ReLU() # activation
        self.fc3 = nn.Linear(9,4)
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out_1 = self.fc1(x)
        out_1 = self.relu1(out_1)
        #out_1 = self.dropout(out_1)
        out_1 = self.fc2(out_1)
        out_1 = self.relu2(out_1)
        out = self.fc3(out_1)

        return out
# Function to save the model
def saveModel():
    path = "./ANNModel.pth"
    torch.save(ANN_model.state_dict(), path)

def fit_model(ANN_model, loss_func, ANN_optimizer, input_shape, num_epochs, train_loader, test_loader):
    # Traning the Model
    #history-like list for store loss & acc value
    best_accuracy = 0.0
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    for epoch in range(num_epochs):
        #training model & store loss & acc / epoch
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            # 1.Define variables
            train = Variable(images.view(input_shape))
            labels = Variable(labels)
            # 2.Clear gradients
            ANN_optimizer.zero_grad()
            # 3.Forward propagation
            outputs = ANN_model(train)
            # 4.Calculate softmax and cross entropy loss
            train_loss = loss_func(outputs, labels)
            # 5.Calculate gradients
            train_loss.backward()
            # 6.Update parameters
            ANN_optimizer.step()
            # 7.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 8.Total number of labels
            total_train += len(labels)
            # 9.Total correct predictions
            correct_train += (predicted == labels).float().sum()
        #10.store val_acc / epoch
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        # 11.store loss / epoch
        training_loss.append(train_loss.data)

        #evaluate model & store loss & acc / epoch
        correct_test = 0
        total_test = 0
        ANN_model.eval()
        for images, labels in test_loader:
            # 1.Define variables
            test = Variable(images.view(input_shape))
            # 2.Forward propagation
            outputs = ANN_model(test)
            # 3.Calculate softmax and cross entropy loss
            val_loss = loss_func(outputs, labels)
            # 4.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 5.Total number of labels
            total_test += len(labels)
            # 6.Total correct predictions
            correct_test += (predicted == labels).float().sum()
        #6.store val_acc / epoch
        val_accuracy = 100 * correct_test / float(total_test)
        validation_accuracy.append(val_accuracy)
        # 11.store val_loss / epoch
        validation_loss.append(val_loss.data)
        print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}% Val_Loss: {} Val_accuracy: {:.6f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
        if val_accuracy > best_accuracy:
            saveModel()
            best_accuracy = val_accuracy
    return training_loss, training_accuracy, validation_loss, validation_accuracy

if __name__ == "__main__":
    import torch
    from torch.autograd import Variable
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms as tfs
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    import random
    import os
    import sys
    import cv2
    sys.path.append('..')

    pic_size = 36
    image_path = './dataset_ann/'

    # 印出dataset中各類有幾張tensor
    for image_count in os.listdir(image_path):
        print(str(len(os.listdir(image_path + image_count))) + " " + image_count + " tensor")

    # 記錄總共有幾張tensor
    file_count = 0
    for floderName in os.listdir(image_path):
        for filename in os.listdir(image_path + floderName):
            file_count +=1
    print('all_tensor_file: ',file_count)

    label_default = np.zeros(shape=[file_count])
    img_default = np.zeros(shape=[file_count,pic_size])
    file_count = 0

    for floderName in os.listdir(image_path):
        for filename in os.listdir(image_path + floderName):
            new_tensor_data=np.load(image_path + floderName + "/" + filename)
            new_tensor_data=np.reshape(new_tensor_data,(-1,36))
            img_default[file_count] = new_tensor_data

            if floderName == '0':
                label_default[file_count] = 0
            elif floderName == '1':
                label_default[file_count] = 1
            elif floderName == '2':
                label_default[file_count] = 2
            elif floderName == '3':
                label_default[file_count] = 3

            file_count +=1

    # reshape成丟進model input的dimension
    img_default = img_default.reshape(file_count,pic_size,1)
    img_default.shape
    #label_onehot=to_categorical(label_default) # 做onehot encoding
    label_onehot=label_default # 不做onehot encoding
    print('label_onehot[0]:{},label_dim:{},shape:{}'.format(label_onehot[0],label_onehot.ndim,label_onehot.shape)) # Label(Encoding結果 , 維度, shape)
    img_default = img_default.astype('float32') / 255.0 # 做 normalization

    random_seed  = 42 # 隨機分割
    features_train, features_test, targets_train, targets_test = train_test_split(img_default, label_onehot, test_size = 0.2, random_state=random_seed) # 切分訓練及測試集
    print('x_train.shape:{}\n,y_train.shape:{}\nx_test.shape:{}\ny_test.shape:{}'.format(features_train.shape, targets_train.shape, features_test.shape, targets_test.shape)) #(train_img, train_label, test_img, test_label)

    featuresTrain = torch.from_numpy(features_train)
    targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

    featuresTest = torch.from_numpy(features_test)
    targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long

    # Pytorch train and test TensorDataset
    train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
    test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

    # Hyper Parameters
    # batch_size, epoch and iteration
    LR = 0.001
    batch_size = 128
    n_iters = 10000
    num_epochs = n_iters / (len(features_train) / batch_size)
    num_epochs = int(num_epochs)
    num_epochs = 100

    # Pytorch DataLoader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)

    #ANN_model = ANN_Model(input_size, hidden_size, num_classes)
    ANN_model = ANN_Model()
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

    print(ANN_model)
    ANN_optimizer = torch.optim.Adam(ANN_model.parameters(), lr=LR, weight_decay=0.001)


    input_shape = (-1,36)

    training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(ANN_model, loss_func, ANN_optimizer, input_shape, num_epochs, train_loader, test_loader)


    # visualization
    plt.plot(range(num_epochs), training_loss, label='Training_loss', color="blue")
    plt.plot(range(num_epochs), validation_loss, label='validation_loss', color="red")
    plt.title('Training & Validation loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./ann_loss.jpg')
    plt.close()

    plt.plot(range(num_epochs), training_accuracy, label='Training_accuracy', color="blue")
    plt.plot(range(num_epochs), validation_accuracy, label='Validation_accuracy', color="red")
    plt.title('Training & Validation accuracy')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./ann_accuracy.jpg')
    plt.close()
