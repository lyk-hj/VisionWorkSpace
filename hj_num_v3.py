import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import AdamW
from torch.autograd import Variable
from model_v3 import Model
import hj_generate_v3 as hj_generate

file_name='2023_3_3_hj_num_1'
new_name='2023_3_3_hj_num_1'
model_path = './weight/'+file_name+'.pt'
save_path = './weight/'+new_name+'.pt'

def show(train_loss,
         train_acc,
         test_acc,
         valid_acc,
         traditional=False):
    if(traditional):
        print("Loss                     || {}\n"
              "Train old Accuracy       || {}\n"
              "Test old Accuracy        || {}\n"
              "Valid old Accuracy       || {}\n".format
              (train_loss,
               train_acc,
               test_acc,
               valid_acc))
    else:
        print("Loss                     || {}\n"
              "Train new Accuracy       || {}\n"
              "Test new Accuracy        || {}\n"
              "Valid new Accuracy       || {}\n".format
              (train_loss,
               train_acc,
               test_acc,
               valid_acc))

def normalized_train(data):
    return data/len(hj_generate.train_dataset)

def normalized_test(data):
    return data/len(hj_generate.test_dataset)

def normalized_valid(data):
    return data/len(hj_generate.valid_dataset)

def train():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = torch.load(model_path, map_location=device)
    # model = Model()
    # model.load_state_dict(torch.load(model_path))
    epochs=10
    lr=0.0001
    loss_fn=CrossEntropyLoss()
    optimizer=AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    # optimizer=SGD(model.parameters(),lr=lr,weight_decay=0.09,momentum=0.99)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    print(model)
    model=model.to(device)

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
            value,pred=torch.max(outputs.data,1)
            optimizer.zero_grad()
            loss=loss_fn(outputs,y_train.long())
            loss.backward()
            optimizer.step()
            value = value*(pred==y_train.data)
            running_loss+=loss.data
            running_correct_old+=torch.sum(pred==y_train.data)
            running_correct_new+=torch.sum(0.2*(pred==y_train.data) + 0.8*value)
        testing_correct_old=0
        testing_correct_new=0
        model.eval()
        for data,label in hj_generate.test_dataloader:
            X_test,y_test=data,label
            X_test,y_test=Variable(X_test).to(device),Variable(y_test).to(device)
            outputs=model(X_test)
            value,pred=torch.max(outputs.data,1)
            value = value*(pred==y_test.data)
            testing_correct_old+=torch.sum(pred==y_test.data)
            testing_correct_new+=torch.sum(0.2*(pred==y_test.data) + 0.8*value)

        valid_correct_old=0
        valid_correct_new=0
        for data,label in hj_generate.valid_dataloader:
            X_check,y_check = data,label
            X_check,y_check = Variable(X_check).to(device),Variable(y_check).to(device)
            outputs = model(X_check)
            value,pred = torch.max(outputs.data,1)
            value = value*(pred==y_check.data)
            valid_correct_old +=torch.sum(pred==y_check.data)
            valid_correct_new+=torch.sum(0.2*(pred==y_check.data) + 0.8*value)

        # obsolete traditional training data
        running_pre_loss = normalized_train(running_loss)
        running_pre_correct_old = normalized_train(running_correct_old)
        testing_pre_correct_old = normalized_test(testing_correct_old)
        valid_pre_correct_old = normalized_valid(valid_correct_old)

        # individual training data
        running_pre_correct_new = normalized_train(running_correct_new)
        testing_pre_correct_new = normalized_test(testing_correct_new)
        valid_pre_correct_new = normalized_valid(valid_correct_new)

        if 0.2*running_pre_correct_new + 0.3*testing_pre_correct_new + 0.5*valid_pre_correct_new > best_accuracy:
            print("more excellent!!!")
            best_accuracy = 0.2*running_pre_correct_new + 0.3*testing_pre_correct_new + 0.5*valid_pre_correct_new
            best_model = model

        show(running_pre_loss,
             running_pre_correct_old,
             testing_pre_correct_old,
             valid_pre_correct_old,
             True)

        show(running_pre_loss,
             running_pre_correct_new,
             testing_pre_correct_new,
             valid_pre_correct_new)


        scheduler.step()
    torch.save(best_model,save_path)
    # torch.onnx.export(best_model, input_data, save_path, opset_version=9, verbose=True, input_names=input_names,
    #                   output_names=output_names, dynamic_axes={'input': {0: '1'}}, )#export model direct


if __name__ == "__main__":
    train()