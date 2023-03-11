import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim import SGD
from torch.autograd import Variable
import hj_generate

file_name='2023_1_31_hj_num_1'
new_name='2023_1_31_hj_num_1'
model_path = './weight/'+file_name+'.pt'
save_path = './weight/'+new_name+'.pt'

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model,self).__init__()
        self.conv=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=0),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=0),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=0),
            torch.nn.MaxPool2d(2),#output 7*12*16 in python is (16,12,7)

            torch.nn.Conv2d(in_channels=32,
                            out_channels=32,
                            kernel_size=3,
                            stride=(2,1),
                            padding=0),#ouput 5*5*32
            torch.nn.AvgPool2d(5),#ouput 1*1*32
            torch.nn.Flatten(),
        )
        self.BN = torch.nn.BatchNorm1d(32)

        self.dense=torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.BatchNorm1d(16),
            torch.nn.Linear(16, 6),
        )
    def forward(self,x):
        x=x.view(-1,1,30,20)#front is rows, back is cols
        x=self.conv(x)
        x=self.BN(x)
        x=self.dense(x)
        return x

def train():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    # model = Model()
    epochs=10
    lr=0.000000000000000000000000000000001

    loss_fn=CrossEntropyLoss()
    optimizer=Adam(model.parameters(),lr=lr,weight_decay=0.02)
    # optimizer=SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    print(model)
    model=model.to(device)
    # state={'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epochs}

    best_model = Model()
    best_accuracy = 0
    for epoch in range(epochs):
        print("Epoch:{}/{}".format(epoch+1,epochs))
        print('-'*10)
        running_loss=0.0
        running_correct_new=0
        running_correct_old=0
        model.train()
        for data,label in hj_generate.train_dataloader:
            X_train,y_train=data,label
            X_train,y_train=Variable(X_train).to(device),Variable(y_train).to(device)
            outputs=model(X_train)
            output = outputs.data
            outputs.data = torch.softmax(outputs.data,1)
            # print(outputs)
            value,pred=torch.max(outputs.data,1)
            _,preds = torch.max(output,1)
            # print(value)
            optimizer.zero_grad()
            loss=loss_fn(outputs,y_train.long())
            # print(outputs)

            loss.backward()
            optimizer.step()
            value = value*(pred==y_train.data)
            # print(value)
            running_loss+=loss.data
            running_correct_old+=torch.sum(preds==y_train.data)
            running_correct_new+=torch.sum(0.2*(pred==y_train.data) + 0.8*value)
            # print(pred==y_train.data)
            # print(0.5*(pred==y_train.data) + 0.5*value)
        testing_correct_old=0
        testing_correct_new=0
        model.eval()
        for data,label in hj_generate.test_dataloader:
            X_test,y_test=data,label
            X_test,y_test=Variable(X_test).to(device),Variable(y_test).to(device)
            outputs=model(X_test)
            outputs.data = torch.softmax(outputs.data, 1)
            value,pred=torch.max(outputs.data,1)
            value = value*(pred==y_test.data)
            testing_correct_old+=torch.sum(pred==y_test.data)
            testing_correct_new+=torch.sum(0.2*(pred==y_test.data) + 0.8*value)
            # print(pred==y_test.data)
            # print(0.5*(pred==y_test.data) + 0.5*value)

        check_correct_old=0
        check_correct_new=0
        for data,label in hj_generate.check_dataloader:
            X_check,y_check = data,label
            X_check,y_check = Variable(X_check).to(device),Variable(y_check).to(device)
            outputs = model(X_check)
            outputs.data = torch.softmax(outputs.data,1)
            value,pred = torch.max(outputs.data,1)
            value = value*(pred==y_check.data)
            check_correct_old +=torch.sum(pred==y_check.data)
            check_correct_new+=torch.sum(0.2*(pred==y_check.data) + 0.8*value)

        if 0.2*running_correct_new + 0.3*testing_correct_new + 0.5*check_correct_new > best_accuracy:
            print("more excellent!!!")
            best_accuracy = 0.2*running_correct_new + 0.3*testing_correct_new + 0.5*check_correct_new
            best_model = model

        print("Loss is {},Train old Accuracy is {},Test old Accuracy is {},Check old Accuracy is {}".format
        (running_loss/len(hj_generate.train_dataset),
         running_correct_old/len(hj_generate.train_dataset),
         testing_correct_old/len(hj_generate.test_dataset),
         check_correct_old/len(hj_generate.check_dataset)))

        print("Loss is {},Train new Accuracy is {},Test new Accuracy is {},Check new Accuracy is {}".format
        (running_loss/len(hj_generate.train_dataset),
         running_correct_new/len(hj_generate.train_dataset),
         testing_correct_new/len(hj_generate.test_dataset),
         check_correct_new/len(hj_generate.check_dataset)))

        scheduler.step()
    torch.save(best_model,save_path)


if __name__ == "__main__":
    train()