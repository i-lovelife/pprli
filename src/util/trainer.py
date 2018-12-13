from src.util.registerable import Registerable
class Trainer(Registerable):
    """
    'trainer':{
    }
    """
    def __init__(self, model):
        self.model = model

@Trainer.register('common')
class CommonTrainer(Trainer):
    """
    {
        'type': 'common',
        'batch_size': 64,
        'train_split': 0.9,
        'epochs': 5
    }
    """
    def __init__(self,
                 batch_size,
                 epochs,
                 model):
        self.batch_size = batch_size
        self.epochs = epochs
        Super(CommonTrainer, self).__init__(model)
    def train(self):
        model = self.model
        model.fit()

