import torch

def aggregator(aggregate_list):
    flag = False
    number = 0
    loss = 0

    for parameter in aggregate_list:
        [model_grad, returned_items, loss_user] = parameter
        del model_grad[14]
        num = len(returned_items)
        loss += loss_user ** 2 * num
        number += num
        if not flag:
            flag = True
            gradient_model = []
            for i in range(len(model_grad)):
                gradient_model.append(model_grad[i] * num)
        else:
            if len(model_grad) != len(gradient_model):
                continue  # 跳过该梯度，因为长度不一致
            for i in range(len(model_grad)):
                gradient_model[i] += model_grad[i] * num
    loss = torch.sqrt(loss / number)
    print('trianing average loss:', loss)
    for i in range(len(gradient_model)):
        gradient_model[i] = gradient_model[i] / number
    return gradient_model