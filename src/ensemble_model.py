from model import SquareModel
import torch
import torch.nn as nn

class Ensemble_Model:
    def __init__(self, checkpoint_paths):
        self.models = []
        for path in checkpoint_paths:
            model = SquareModel()
            model.load_state_dict(torch.load(path))
            self.models.append(model)
        self.softmax = nn.Softmax(dim=1)

    def __call__(self, x):
        pred = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                output = self.softmax(output)
                pred.append(output)
        mean_pred = torch.mean(torch.stack(pred), dim=0)
        argmax = torch.argmax(mean_pred, dim=1).numpy()
        return argmax


# unit test
if __name__ == '__main__':
    import os
    model_dir = './output'
    model1 = os.path.join(model_dir, 'cnn_model_fold=0_epoch=7_loss=0.137.pth')
    model2 = os.path.join(model_dir, 'cnn_model_fold=0_epoch=16_loss=0.097.pth')
    model = Ensemble_Model([model1, model2])
    x = torch.randn(1, 3, 128, 128)
    pred = model(x)
    print(pred)