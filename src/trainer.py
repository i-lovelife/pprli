from src.util.registerable import Registerable
from keras.callbacks import Callback
from src.callbacks import EvaluaterCallback

class Trainer(Registerable):
    _default_type='keras'
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
    def train(self, datasaet, worker=None, evaluaters=None, callbacks=[]):
        raise NotImplementedError

class IterTrainer(Trainer):
    def __init__(self,
                 total_iter=5000,
                 log_iter=10,
                 evaluate_iter=500,
                 **args):
        super().__init__(**args)
        self.total_iter = total_iter
        self.log_iter = log_iter
        self.evaluate_iter = evaluater_iter

@Trainer.register('adv')
class AdversarialTrainer(IterTrainer):
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
              evaluaters=None,
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
        
        evaluater:
            evaluate(dataset, worker): evaluate the worker on dataset
        """
        evaluate_history = []
        if worker is None:
            return
        if evaluaters is not None:
            evaluaters = list(evaluaters)
        for it in range(self.total_iter):
            for j in range(self.d_iter):
                data = dataset.get_train_batch(self.batch_size)
                output_d = worker.d_train_model.train_on_batch(*worker.get_input_d(data))
            for j in range(self.g_iter):
                data = dataset.get_train_batch(self.batch_size)
                output_g = worker.g_train_model.train_on_batch(*worker.get_input_g(data))
            if it % self.log_iter == 0:
                print(f'iter {it}: g_log={output_g} d_log={output_d}')
            if it % self.evaluate_iter == 0 and evaluaters is not None:
                output_evaluater = [evaluater.evaluate(dataset, worker) for evaluater in evaluaters]
                evaluate_history.append(output_evaluater)
                print(f'iter {it}: eva={output_evaluater}')
                
        return evaluate_history

@Trainer.register('keras')
class KerasTrainer(Trainer):
    def __init__(self,
                 epochs=5,
                 **args
                ):
        super().__init__(**args)
        self.epochs = epochs

    def train(self, dataset, worker, evaluaters=None, callbacks=[]):
        """
        dataset:
            get_train(): get all train data
            get_test(): get all test data
        model:
            train(dataset, epochs, batch_size): train model in dataset for epochs
        """
        train_data = dataset.get_train()
        test_data = dataset.get_test()
        #import pdb;pdb.set_trace()
        if evaluaters is not None:
            evaluater_callback = EvaluaterCallback(evaluaters, dataset, worker)
            callbacks.append(evaluater_callback)
        history = worker.train_model.fit(*worker.get_input(train_data),
                                         epochs=self.epochs, 
                                         batch_size=self.batch_size,
                                         validation_data=worker.get_input(test_data),
                                         callbacks=callbacks)
        if evaluaters is not None:
            return evaluater_callback.evaluate_history
        return history