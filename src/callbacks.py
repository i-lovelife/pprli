from keras.callbacks import Callback as KerasCallback

class Callback(KerasCallback):
    def get_output_history(self):
        pass
    
class EvaluaterCallback(Callback):
    def __init__(self, evaluater, dataset, worker):
        self.evaluater = evaluater
        self.dataset = dataset
        self.worker = worker
        self.output_history = []
    def on_train_begin(self, logs=None):
        self.output_history = []
    def on_epoch_end(self, epoch, logs=None):
        evaluater = self.evaluater
        dataset = self.dataset
        worker = self.worker
        output = evaluater.evaluate(dataset, 
                                    worker,
                                    epoch)
        if output is not None:
            print(f'epoch {epoch}: output {type(self.evaluater).__name__} = {output}')
            self.output_history.append(output)
    def get_output_history(self):
        return self.output_history

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