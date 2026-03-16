import copy
import torch.nn as nn
import functions as fn

#參數
model_name = 'lstm'
lookback = 6
predict_length = 6
batch_size = 1024
epoch = 2000


with open('test29', 'w') as file:
    
    train, valid, test = fn.read_data()
    park_names = train.columns
    dataset_dict, data_loader_dict = fn.get_dataloader(train, valid, test, batch_size, lookback, predict_length)
    model_dict, optimizer_dict = fn.get_model_optim(park_names, model_name, lookback)
    loss_function = nn.MSELoss()

    #訓練與驗證
    for i in range(len(park_names)):
        file.write(f'Model = {model_name}, Training in {park_names[i]} | model = {model_name}\n')

        test_th = 100
        test_model = copy.deepcopy(model_dict[model_name, park_names[i]])
        for e in range(epoch):

            #訓練集
            model_dict[model_name, park_names[i]].train()
            for j, batch in enumerate(data_loader_dict['train', park_names[i]]):
                sample, label = batch
                optimizer_dict[model_name, park_names[i]].zero_grad()
                predict = model_dict[model_name, park_names[i]](sample)
                loss = loss_function(predict, label)
                loss.backward()
                optimizer_dict[model_name, park_names[i]].step()

            #驗證集
            model_dict[model_name, park_names[i]].eval()
            for j, batch in enumerate(data_loader_dict['valid', park_names[i]]):
                sample, label = batch
                predict = model_dict[model_name, park_names[i]](sample)
                valid_loss = loss_function(predict, label)

                if valid_loss < test_th:
                    test_th = valid_loss
                    test_model = copy.deepcopy(model_dict[model_name, park_names[i]])

            if (e+1) % 10 == 0:
                file.write(f'EPOCH = {e+1} / {epoch} | train loss = {loss.item()} | valid loss = {valid_loss.item()}\n')

        # 測試集
        for j, batch in enumerate(data_loader_dict['test', park_names[i]]):
            sample, label = batch
            test_predict = test_model(sample)
            test_loss = loss_function(test_predict, label)
            np_predict = test_predict.detach().numpy()
            np_label = label.detach().numpy()
            mse, rmse, mae, mape, r2 = fn.calculate_metrics(np_predict, np_label)
            # 結果
            file.write(f'Test in {park_names[i]} | model = {model_name}\n')
            file.write(f'Predicted vacancy rate: {np_predict}\n')
            file.write(f'MSE = {mse}, RMSE = {rmse}, MAE = {mae}, MAPE = {mape}, R-square = {r2}\n')
