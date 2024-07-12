from src.utils.context import Context
import optuna

class HpTuner():

    def __init__(self, context: Context, n_trials: int) -> None:
        self.context = context
        self.n_trials = n_trials
        self.dataset = context.factories['datasets'].get_dataset(context.conf['do-pairs'][0]['dataset'])

        do_pairs_list = context.conf['do-pairs']
        self.dataset_config = do_pairs_list[0]['dataset']
        self.oracle_config = do_pairs_list[0]['oracle']


    def optimize(self):
        #Create a study object and optimize the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Print the best parameters found
        print('Best parameters:', study.best_params)
        print('Best value:', study.best_value)

    #Define an objective function to be maximized.
    def objective(self, trial):

        #Suggest values of the hyperparameters using a trial object.
        #batch_size = trial.trial.suggest_categorical('batch_size', [32, 64])
        learning_rate = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        #weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
        num_conv_layers = trial.suggest_int('num_conv_layers', 1, 6)
        num_dense_layers = trial.suggest_int('num_dense_layers', 1, 6)
        conv_booster = trial.suggest_int('conv_booster', 1, 6)
        linear_decay = trial.suggest_float('linear_decay', 1e-1, 3)

        self.oracle_config['parameters']['optimizer']['parameters']['lr'] = learning_rate
        #self.oracle_config['parameters']['optimizer']['parameters']['weight_decay'] = weight_decay

        self.oracle_config['parameters']['model']['num_conv_layers'] = num_conv_layers
        self.oracle_config['parameters']['model']['num_dense_layers'] = num_dense_layers
        self.oracle_config['parameters']['model']['conv_booster'] = conv_booster
        self.oracle_config['parameters']['model']['linear_decay'] = linear_decay

        dataset = self.context.factories['datasets'].get_dataset(self.dataset_config)
        oracle = self.context.factories['oracles'].get_oracle(self.oracle_config, dataset)

        accuracy = oracle.evaluate(dataset, fold_id=-1)
        return accuracy