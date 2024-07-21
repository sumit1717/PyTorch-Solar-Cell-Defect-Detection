import os

import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        self._optim.zero_grad()

        outputs = self._model(x)
        loss = self._crit(outputs, y)
        loss.backward()
        self._optim.step()

        return loss.item()
        
    def val_test_step(self, x, y):

        outputs = self._model(x)
        loss = self._crit(outputs, y)
        return loss.item(), outputs
        
    def train_epoch(self):
        self._model.train()
        running_loss = 0.0

        for batch in self._train_dl:

            inputs, targets = batch['image'], batch['label']
            if self._cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            loss = self.train_step(inputs, targets)
            running_loss += loss * inputs.size(0)

        epoch_loss = running_loss / len(self._train_dl.dataset)
        return epoch_loss
    
    def val_test(self):
        self._model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []

        with t.no_grad():
            for batch in self._val_test_dl:

                inputs, targets = batch['image'], batch['label']
                if self._cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                loss, outputs = self.val_test_step(inputs, targets)

                running_loss += loss * inputs.size(0)
                all_predictions.append(outputs)
                all_targets.append(targets)

        # Calculate the average loss and average metrics of your choice
        epoch_loss = running_loss / len(self._val_test_dl.dataset)
        all_predictions = t.cat(all_predictions)
        all_targets = t.cat(all_targets)

        predictions_np = (all_predictions > 0.5).cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        f1 = f1_score(targets_np, predictions_np, average='macro')

        return epoch_loss, f1
        
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        train_losses = []
        val_losses = []
        val_f1_scores = []

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_f1 = self.val_test()
            t.cuda.empty_cache()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1_scores.append(val_f1)

            print(
                f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation F1: {val_f1:.4f}')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch)
            else:
                self.epochs_without_improvement += 1

            if 0 < self._early_stopping_patience <= self.epochs_without_improvement:
                print("Early stopping triggered")
                break

        return train_losses, val_losses, val_f1_scores
                    
        
        
        
