from keras.callbacks import Callback as KerasCallback

class Callback(KerasCallback):
    pass
    
class EvaluaterCallback(Callback):
    def __init__(self, evaluaters, dataset, worker):
        self.evaluaters = evaluaters
        self.dataset = dataset
        self.worker = worker
        self.evaluate_history = []
    def on_train_begin(self, logs=None):
        self.evaluate_history = []
    def on_epoch_end(self, epoch, logs=None):
        evaluaters = self.evaluaters
        dataset = self.dataset
        worker = self.worker
        output_evaluater = [evaluater.evaluate(dataset, worker) for evaluater in evaluaters]
        self.evaluate_history.append(output_evaluater)
        print(f'epoch {epoch}: eva={output_evaluater}')

class EarlyStopping(Callback):
    def __init__(self,
                 ideal_acc=0.9,
                 verbose=True):
        super().__init__()
        self.ideal_acc = ideal_acc
        self.verbose = verbose
        
    def on_train_begin(self, logs=None):
        self.best_val_acc = 0
        self.best_acc = 0
        self.stopped_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        assert 'val_acc' in logs
        assert 'acc' in logs
        val_acc, acc = logs.get('val_acc'), logs.get('acc')
        self.best_val_acc = max(val_acc, self.best_val_acc)
        self.best_acc = max(acc, self.best_acc)
        if min(self.best_acc, self.best_val_acc) > self.ideal_acc:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch >= 0 and self.verbose is True:
            print(f'Stopped at Epoch {self.stopped_epoch+1}: acc={self.best_acc} val_acc={self.best_val_acc}')