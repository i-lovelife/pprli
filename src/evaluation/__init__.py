from src.util.registerable import Registerable
class Evaluator(Registerable):
    pass

@Evaluator.register('mi')
class MiEvaluator(Evaluator):
    def __init__(self):
        pass
    
    def evaluate(self):
       pass 


