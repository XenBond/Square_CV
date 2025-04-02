from data import SquareDataset
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SquareModel
from tqdm.auto import tqdm

# random seed fixing
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def CV_split(df, n_folds=5):
    '''
    split the dataframe into n_folds
    '''
    # random shuffle and assign n_folds
    df = df.sample(frac=1).reset_index(drop=True)
    df['fold'] = df.index % n_folds
    
    # return 5-fold's train and validation set
    folds = {}
    for i in range(n_folds):
        train = df[df['fold'] != i]
        val = df[df['fold'] == i]
        folds[i] = (train, val)
    return folds

def train_model(
        model, 
        train_dataset, 
        val_dataset, 
        learning_rate=0.001,
        n_epochs=10,
        batch_size=32,
        device='cuda',
        model_name='model',
        output_dir='.',
    ):
    model.train()
    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model.to(device)
    print(f'Training {model_name}...')
    for epoch in range(n_epochs):
        model.train()
        for i, data in tqdm(enumerate(train_loader), desc=f'Epoch {epoch}'):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            all_loss = []
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels).cpu().numpy().tolist()
                all_loss.extend(loss)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            total_loss = sum(all_loss) / len(all_loss)
            print(f'Epoch {epoch}, Accuracy: {accuracy}, Loss: {total_loss}')
            torch.save(model.state_dict(), f'{output_dir}/{model_name}_epoch={epoch}_loss={total_loss:.3f}.pth')
    del model, criterion, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()

if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='cnn_model')
    parser.add_argument('--training_data', type=str, default='/home/fang/Square_Task/squares/train')
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./output')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = SquareModel()
    training_data = '/home/fang/Square_Task/squares/train'
    
    # get the pandas dataframe
    dataset_ = SquareDataset(training_data)
    df = dataset_.df

    folds = CV_split(df)

    # mean and std for normalization, from EDA.py
    mean = [0.5075626373291016, 0.5077584385871887, 0.5012454986572266]
    std=[0.24075263738632202, 0.24078987538814545, 0.21087391674518585]

    for i in range(5):
        train_df, val_df = folds[i]
        train_df.to_csv(f'{args.output_dir}/train_fold_{i}.csv', index=False)
        val_df.to_csv(f'{args.output_dir}/val_fold_{i}.csv', index=False)
        train_dataset = SquareDataset(training_data, mean=mean, std=std)
        train_dataset.df = train_df
        val_dataset = SquareDataset(training_data, mean=mean, std=std)
        val_dataset.df = val_df
        print(f'Start training fold {i}...')
        train_model(
            model, 
            train_dataset, 
            val_dataset, 
            model_name=args.model_name + '_fold=' + str(i), 
            n_epochs=args.n_epochs, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            device=args.device,
            output_dir=args.output_dir
        )
