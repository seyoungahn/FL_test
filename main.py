import logging
from cnn import CNN
import datautils
import utils
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def train(model, optimizer, criterion, trainloader, metrics, params):
    model.train()

    summaries = []
    avg_loss = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(trainloader) * params.t_local_epochs, position=0, leave=True) as t:
        for loc_epoch in range(params.t_local_epochs):
            for batch_idx, (x_train, y_train) in enumerate(trainloader):
                # Move to GPU if available
                if params.t_cuda:
                    x_train, y_train = x_train.cuda(non_blocking=True), y_train.cuda(non_blocking=True)

                x_train, y_train = torch.autograd.Variable(x_train), torch.autograd.Variable(y_train)

                y_pred = model(x_train)
                loss = criterion(y_pred, y_train)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Evaluation (train acc, train loss)
                if batch_idx % params.t_save_summary_steps == 0:
                    y_pred = y_pred.data.cpu().numpy()
                    y_train = y_train.data.cpu().numpy()

                    summary = {metric: metrics[metric](y_pred, y_train) for metric in metrics}
                    summary['loss'] = loss.item()
                    summaries.append(summary)

                avg_loss.update(loss.item())

                t.set_postfix(loss='{:05.3f}'.format(avg_loss()))
                t.update()
            # print(summaries)
            metrics_mean = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]}
        return metrics_mean

def evaluate(model, criterion, validloader, metrics, params):
    """
    Evaluate the model on 'num_steps' batches
    :param model: (torch.nn.Module) the neural network
    :param criterion: a function that takes y_pred and y_valid and computes the loss for the batch
    :param validloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
    :param metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
    :param params: (Params) hyperparameters
    :return:
    """
    # Set model to evaluation mode
    model.eval()

    # Summary for current eval loop
    summaries = []

    # Compute metrics over the dataset
    for batch_idx, (x_valid, y_valid) in enumerate(validloader):
        # Move to GPU if available
        if params.t_cuda:
            x_valid, y_valid = x_valid.cuda(non_blocking=True), y_valid.cuda(non_blocking=True)
        # Fetch the next evaluation batch
        x_valid, y_valid = torch.autograd.Variable(x_valid), torch.autograd.Variable(y_valid)

        # Compute model output
        y_pred = model(x_valid)
        loss = criterion(y_pred, y_valid)

        # Extract data from torch Variable, move to CPU, convert to numpy arrays
        y_pred = y_pred.data.cpu().numpy()
        y_valid = y_valid.data.cpu().numpy()

        # Compute all metrics on this batch
        summary = {metric: metrics[metric](y_pred, y_valid) for metric in metrics}
        summary['loss'] = loss.item()
        summaries.append(summary)

    # Compute mean of all metrics in summary
    # print("Evaluation summaries")
    # print(summaries)
    metrics_mean = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics: " + metrics_string)
    return metrics_mean

def FedAvg(global_model, models, data_sizes):
    # Objective: aggregates all local model to the global model
    # Inputs: global model, a list of secondary UEs, experiment parameters
    # Outputs: parameter dictionary of aggregated model
    global_model_dict = dict(global_model.state_dict())
    aggregated_dict = dict(global_model.state_dict())
    parties_dict = {}
    for i in range(len(models)):
        parties_dict[i] = dict(models[i].state_dict())
    beta = data_sizes / sum(data_sizes)
    print("Data_sizes: ", data_sizes)
    print("Sum: {}".format(sum(data_sizes)))
    print("Beta: ", beta)
    for name, param in global_model_dict.items():
        aggregated_dict[name].data.copy_(sum([beta[i] * parties_dict[i][name].data for i in range(len(models))]))
    return aggregated_dict

# Hyperparameter setting
json_path = os.path.join("params.json")
assert os.path.isfile(json_path), "No JSON configuration file found at {}".format(json_path)
params = utils.Params(json_path)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

utils.set_logger(os.path.join('train.log'))

## Set non-IID dataset based on Dirichlet distribution
logging.info("+ Loading the non-IID datasets based on Dirichlet distributions ({}, alpha={})...".format(params.t_dataset_type, params.alpha))
trainloader_nIID, testloader_nIID, DSI_list = datautils.fetch_noniid_dirichlet_dataloader(params)
data_sizes = []
for i in range(params.n_users):
    data_sizes.append(len(trainloader_nIID[i].dataset))
data_sizes = np.array(data_sizes)
print(data_sizes)
logging.info(" done.")

## Set IID dataset
logging.info("+ Loading the IID dataset ...".format(params.t_dataset_type, params.alpha))
trainloader_IID, testloader_IID = datautils.fetch_dataloader(params)
print("Data_size: {}".format(len(trainloader_IID.dataset)))
logging.info(" done.")

## Set initial configurations for FL
models = []
optimizers = []
global_model = CNN().to(device) if params.t_cuda else CNN()
for i in range(params.n_users):
    model = CNN().to(device) if params.t_cuda else CNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    models.append(model)
    optimizers.append(optimizer)

## Set initial configurations for centralized learning
centralized_model = CNN().to(device) if params.t_cuda else CNN()
centralized_optim = optim.SGD(centralized_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
diff_model = CNN().to(device) if params.t_cuda else CNN()

for epoch in tqdm(range(100), desc='EPOCHS'):
    logging.info("EPOCH #{}".format(epoch + 1))
    ## Federated learning
    logging.info("+ FEDERATED LEARNING")
    for i in range(params.n_users):
        models[i].load_state_dict(global_model.state_dict())
        logging.info("  - LOCAL TRAINING (user #{})...".format(i+1))
        train_metrics = train(models[i], optimizers[i], utils.loss_function, trainloader_nIID[i], utils.metrics, params)

    logging.info("  - GLOBAL AGGREGATION")
    aggregated_dict = FedAvg(global_model, models, data_sizes)
    global_model.load_state_dict(aggregated_dict)
    global_valid_metrics = evaluate(global_model, utils.loss_function, testloader_nIID, utils.metrics, params)
    global_acc = global_valid_metrics['accuracy']
    global_loss = global_valid_metrics['loss']
    logging.info("  - FL PERFORMANCE - acc: {}, loss: {}".format(global_acc, global_loss))

    ## Centralized learning
    logging.info("+ CENTRALIZED LEARNING")
    train_metrics = train(centralized_model, centralized_optim, utils.loss_function, trainloader_IID, utils.metrics, params)
    centralized_valid_metrics = evaluate(centralized_model, utils.loss_function, testloader_IID, utils.metrics, params)
    centralized_acc = centralized_valid_metrics['accuracy']
    centralized_loss = centralized_valid_metrics['loss']
    logging.info("  - CL PERFORMANCE - acc: {}, loss: {}".format(centralized_acc, centralized_loss))

    ## Weight difference
    wd = sum(torch.norm(p - q, 2) for p, q in zip(global_model.parameters(), centralized_model.parameters()))
    logging.info("Weight difference: {}".format(wd.item()))
    utils.write_csv(".", "perf", [epoch+1, global_acc, global_loss, centralized_acc, centralized_loss, wd.item()], ['Epochs', 'Acc(global)', 'Loss(global)', 'Acc(central)', 'Loss(central)', 'Weight difference'])
