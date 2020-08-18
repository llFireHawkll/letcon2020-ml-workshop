'''
File: optuna.py
Project: optimizer
File Created: Monday, 17th August 2020 3:53:01 pm
Author: Sparsh Dutta (sparsh.dtt@gmail.com)
-----
Last Modified: Monday, 17th August 2020 5:14:39 pm
Modified By: Sparsh Dutta (sparsh.dtt@gmail.com>)
-----
Copyright 2020 Sparsh Dutta
'''
import optuna

__all__ = ['OptunaOptimizer']

class OptunaOptimizer:
    def __init__(self, config):
        self.seed = config['seed']
        self.model_to_optimize = config['model']
        self.scoring_function = config['score_function']
        self.fixed_params = config['params']['fixed']
        self.varying_params = config['params']['varying']   

        self.optimizer()

    def optimizer(self):
        sampler = optuna.samplers.TPESampler(seed=self.seed)    
        self.study = optuna.create_study(direction='maximize', sampler=sampler)
        self.study.optimize(self.objective, n_trials=self.fixed_params['n_trials'])
        
        self.best_params = self.study.best_params
    
    def objective(self, trial):
        params = dict()
        params['nthread'] = -1

        for parameter in self.varying_params.keys():
            
            if type(self.varying_params[parameter]) == list:
                params[parameter] = trial.suggest_categorical(parameter,
                                                             self.varying_params[parameter])
            elif self.varying_params[parameter].get('step'):
                min_value = self.varying_params[parameter]['min']
                max_value = self.varying_params[parameter]['max']
                step_value = self.varying_params[parameter]['step']

                params[parameter] = trial.suggest_discrete_uniform(parameter,
                                                                   min_value,
                                                                   max_value,
                                                                   step_value)
            else:
                min_value = self.varying_params[parameter]['min']
                max_value = self.varying_params[parameter]['max']

                if type(min_value) == int:
                    params[parameter] = trial.suggest_int(parameter,
                                                         min_value,
                                                         max_value)                                    
                else:
                    params[parameter] = trial.suggest_loguniform(parameter,
                                                                min_value,
                                                                max_value)

        model = self.model_to_optimize(params=params)
        _, valid_score = self.scoring_function(model=model, 
                                              return_score=True)
        return valid_score