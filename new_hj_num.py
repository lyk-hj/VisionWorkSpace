import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import AdamW
from torch.autograd import Variable
from model import Model
# import hj_generate_v2 as hj_generate
import hj_generate_v2 as hj_generate

file_name='2023_2_16_hj_num_1'
new_name='2023_2_16_hj_num_1'
model_path = './weight/'+file_name+'.pt'
save_path = './weight/'+new_name+'.pt'


def train():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = torch.load(model_path, map_location=device)
    # model = Model()
    # model.load_state_dict(torch.load(model_path))
    epochs=10
    lr=0.00000001
    loss_fn=CrossEntropyLoss()
    optimizer=AdamW(model.parameters(),lr=lr,weight_decay=0.0005)
    # optimizer=SGD(model.parameters(),lr=lr,weight_decay=0.09,momentum=0.99)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
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
            # print(outputs)
            value,pred=torch.max(outputs.data,1)
            optimizer.zero_grad()
            loss=loss_fn(outputs,y_train.long())
            loss.backward()
            optimizer.step()
            value = value*(pred==y_train.data)
            # print(value)
            running_loss+=loss.data
            running_correct_old+=torch.sum(pred==y_train.data)
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
            # outputs.data = torch.softmax(outputs.data, 1)
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
            value,pred = torch.max(outputs.data,1)
            value = value*(pred==y_check.data)
            check_correct_old +=torch.sum(pred==y_check.data)
            check_correct_new+=torch.sum(0.2*(pred==y_check.data) + 0.8*value)

        # obsolete traditional training data
        running_pre_loss = running_loss/len(hj_generate.train_dataset)
        running_pre_correct_old = running_correct_old/len(hj_generate.train_dataset)
        testing_pre_correct_old = testing_correct_old/len(hj_generate.test_dataset)
        check_pre_correct_old = check_correct_old / len(hj_generate.check_dataset)

        # individual training data
        running_pre_correct_new = running_correct_new/len(hj_generate.train_dataset)
        testing_pre_correct_new = testing_correct_new/len(hj_generate.test_dataset)
        check_pre_correct_new = check_correct_new/len(hj_generate.check_dataset)
        if 0.2*running_pre_correct_new + 0.3*testing_pre_correct_new + 0.5*check_pre_correct_new > best_accuracy:
            print("more excellent!!!")
            best_accuracy = 0.2*running_pre_correct_new + 0.3*testing_pre_correct_new + 0.5*check_pre_correct_new
            # print(model.parameters())
            best_model = model

        print("Loss is {},Train old Accuracy is {},Test old Accuracy is {},Check old Accuracy is {}".format
        (running_pre_loss,
         running_pre_correct_old,
         testing_pre_correct_old,
         check_pre_correct_old))

        print("Loss is {},Train new Accuracy is {},Test new Accuracy is {},Check new Accuracy is {}".format
        (running_pre_loss,
         running_pre_correct_new,
         testing_pre_correct_new,
         check_pre_correct_new))
        # print(len(hj_generate.train_dataset))
        # print(running_correct_old)

        scheduler.step()
    torch.save(best_model,save_path)
    # torch.onnx.export(best_model, input_data, save_path, opset_version=9, verbose=True, input_names=input_names,
    #                   output_names=output_names, dynamic_axes={'input': {0: '1'}}, )#export model direct


if __name__ == "__main__":
    train()