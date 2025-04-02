import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import torch.utils.data as data
import pandas as pd
from model import SquareModel
from ensemble_model import Ensemble_Model
from data import SquareDataset
from tqdm.auto import tqdm

class Evaluator:
    def __init__(
        self,
        model,
        dataset,
    ):
        self.model = model
        self.dataset = dataset

    def evaluate(self):
        y_true = []
        y_pred = []
        error = {}
        for i in tqdm(range(len(self.dataset)), desc='Evaluating'):
            img, label = self.dataset[i]
            pred = self.model(img.unsqueeze(0))
            y_true.append(label)
            y_pred.append(pred)
            if label != pred:
                error[i] = (img, label, pred)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        # save the error images
        error_dir = './error_images'
        if not os.path.exists(error_dir):
            os.makedirs(error_dir)
            for i in range(3):
                os.makedirs(os.path.join(error_dir, str(i)))
        for i, (img, label, pred) in error.items():
            img = img.numpy().transpose(1, 2, 0) * 255
            img = Image.fromarray(img.astype(np.uint8))
            img.save(os.path.join(error_dir, str(label), f'{i}_pred={pred}.png'))
        
        # print the confusion matrix and precision/recall
        confusion_matrix = self.confusion_matrix(y_true, y_pred)
        individual_precision, individual_recall, precision, recall, F_1 = self.precision_recall(y_true, y_pred)
        print(f'Confusion Matrix: {confusion_matrix}')
        print(f'Individual Precision: {individual_precision}')
        print(f'Individual Recall: {individual_recall}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F_1: {F_1}')
        
    def confusion_matrix(self, y_true, y_pred):
        cm = metrics.confusion_matrix(y_true, y_pred)
        return cm
    
    def precision_recall(self, y_true, y_pred):
        individual_precision = metrics.precision_score(y_true, y_pred, average=None)
        individual_recall = metrics.recall_score(y_true, y_pred, average=None)
        precision = metrics.precision_score(y_true, y_pred, average='macro')
        recall = metrics.recall_score(y_true, y_pred, average='macro')
        F_1 = metrics.f1_score(y_true, y_pred, average='macro')
        return individual_precision, individual_recall, precision, recall, F_1
    
if __name__ == '__main__':
    import os
    mean = [0.5075626373291016, 0.5077584385871887, 0.5012454986572266]
    std=[0.24075263738632202, 0.24078987538814545, 0.21087391674518585]
    test_dataset = SquareDataset('/home/fang/Square_Task/squares/val', mean=mean, std=std)
    # select best models:
    model_dir = './output'
    all_models = os.listdir(model_dir)
    best_models = {
        'fold=0': [],
        'fold=1': [],
        'fold=2': [],
        'fold=3': [],
        'fold=4': [],
    }
    for model in all_models:
        if 'cnn_model' in model:
            fold = model.split('_')[2]
            loss = float(model.split('_')[4].split('=')[1].split('.pt')[0])
            best_models[fold].append((model, loss))
    best_models = {k: sorted(v, key=lambda x: x[1]) for k, v in best_models.items()}
    best_models = {k: os.path.join(model_dir, v[0][0]) for k, v in best_models.items()}
    print('Best Models:')
    for k, v in best_models.items():
        print(f'Fold {k}: {v}')
    model = Ensemble_Model([v for k, v in best_models.items()])
    evaluator = Evaluator(model, test_dataset)
    evaluator.evaluate()
