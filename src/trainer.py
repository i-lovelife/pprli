from src.util.registerable import Registerable
from keras.callbacks import Callback
from src.callbacks import EvaluaterCallback, ModelCheckpoint

def collect_history(callbacks):
    history = {}
    for idx, callback in enumerate(callbacks):
        output = callback.get_output_history()
        if output is not None:
            key, value = output
            if value != []:
                history[f'{key}_{idx}'] = value
    return history

class Trainer(Registerable):
    _default_type='keras'
    def __init__(self, epochs=5, batch_size=128, verbose=True, save_dir=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.save_dir = save_dir
    def train(self, datasaet, worker=None, evaluaters=None, callbacks=[]):
        raise NotImplementedError

@Trainer.register('adv')
class AdversarialTrainer(Trainer):
    def __init__(self,
                 d_iter=2,
                 g_iter=1,
                 **args):
        super().__init__(**args)
        self.d_iter = d_iter
        self.g_iter = g_iter
    def train(self,
              dataset,
              worker=None,
              callbacks=[]):
        """
        dataset:
            get_train_batch(batch_size): get a batch of training data
            get_test_batch(batch_size): get a batch of test data
            get_train(): get all train data
            get_test(): get all test data

        worker:
            train_d(data): train worker d on data for one iter
            train_g(data): train worker g on data for one iter
            predict(data): output encrypted version of data
        
        callbacks:
            it can process callback that don't need logs
        """
        iters_per_epoch = dataset.get_train_len() // self.batch_size
        if worker is None:
            return
        for callback in callbacks:
            callback.on_train_begin(worker)
        for epoch in range(self.epochs):
            for iter_no in range(iters_per_epoch):
                for j in range(self.d_iter):
                    data1 = dataset.get_train_batch(self.batch_size)
                    data2 = dataset.get_train_batch(self.batch_size)
                    output_d = worker.d_train_model.train_on_batch(*worker.get_input_d(data1, data2))
                for j in range(self.g_iter):
                    data1 = dataset.get_train_batch(self.batch_size)
                    data2 = dataset.get_train_batch(self.batch_size)
                    output_g = worker.g_train_model.train_on_batch(*worker.get_input_g(data1, data2))
                print(f'iter {iter_no}: g_log={output_g} d_log={output_d}')
            epoch_history = {}
            logs = {}
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)
            if self.save_dir is not None:
                worker.save_weights(self.save_dir / f'model_weight_{epoch}.hdf5')
        for callback in callbacks:
            callback.on_train_end()
            
        history = collect_history(callbacks)
        
        return history

@Trainer.register('keras')
class KerasTrainer(Trainer):
    def train(self, dataset, worker, callbacks=[]):
        """
        dataset:
            get_train(): get all train data
            get_test(): get all test data
        model:
            train(dataset, epochs, batch_size): train model in dataset for epochs
        """
        if self.save_dir is not None:
            callbacks.append(ModelCheckpoint(str(self.save_dir/'model_weight_{epoch}.hdf5'), save_best_only=False, save_weights_only=True))
        train_data = dataset.get_train()
        test_data = dataset.get_test()
        #import pdb;pdb.set_trace()
        worker.train_model.fit(*worker.get_input(train_data),
                               epochs=self.epochs, 
                               batch_size=self.batch_size,
                               validation_data=worker.get_input(test_data),
                               verbose= self.verbose,
                               callbacks=callbacks)
        
        history = collect_history(callbacks)
        return history