from src.util.registerable import Registerable
class Trainer(Registerable):
    pass

@Trainer.register('privater')
class PrivaterTrainer(Trainer):
    def __init__(self,
                 batch_size=64,
                 total_iter=5000,
                 log_iter=10,
                 evaluate_iter=500,
                 d_iter=2,
                 g_iter=1
                 **args):
        Super(PrivaterTrainer, self).__init__(**args)
        self.batch_size = batch_size
        self.total_iter = total_iter
        self.log_iter = log_iter
        self.evaluate_iter = evaluate_iter
        self.d_iter = d_iter
        self.g_iter = g_iter
    def train(self,
              dataset,
              privater=None,
              evaluater=None):
        """
        dataset:
            get_train_batch(batch_size): get a batch of training data
            get_test_batch(batch_size): get a batch of test data
            get_train(): get all train data
            get_test(): get all test data

        privater:
            train_d(dataset, batch_size): train d_train_model of privater for a single iter
            train_g(dataset, batch_size): train g_train_model of privater for a single iter
            predict(data): output encrypted version of data
        
        evaluater:
            evaluate(dataset, privater): evaluate the privater on dataset
        """
        if privater is None:
            return
        for it in range(self.total_iter):
            for j in range(self.d_iter):
                output_d = privater.train_d(dataset, batch_data)
            for j in range(self.g_iter):
                output_g = privater.train_g(dataset, batch_data)
            if it % self.log_iter == 0:
                print(f'iter {it}: g_log={output_g} d_log={output_d}')
            if it % self.evaluate_iter == 0 and evaluater is not None:
                output_evaluater = evaluater.evaluate(dataset, privater)
                print(f'iter {it}: eva_log={output_evaluater}')

@Trainer.register('common')
def CommonTrainer(Trainer):
    def __init__(self,
                 batch_size=128,
                 epochs=5,
                 **args):
        Super(CommonTrainer, self).__init__(**args)
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, dataset, model):
        """
        dataset:
            get_train(): get all train data
            get_test(): get all test data
        model:
            train(dataset, epochs, batch_size): train model in dataset for epochs
        """
        model.train(dataset, epochs=self.epochs, batch_size=self.batch_size)
