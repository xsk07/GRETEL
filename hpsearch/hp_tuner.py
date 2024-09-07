from src.utils.context import Context
import optuna

class HpTuner():

    def __init__(self, context: Context, train_oracle: bool, train_expainer: bool, n_trials: int) -> None:
        self.context = context
        self.n_trials = n_trials
        self.dataset = context.factories['datasets'].get_dataset(context.conf['do-pairs'][0]['dataset'])

        do_pairs_list = context.conf['do-pairs']
        self.dataset_config = do_pairs_list[0]['dataset']
        self.oracle_config = do_pairs_list[0]['oracle']

        optuna.logging.set_verbosity(optuna.logging.INFO)
        optuna.logging.disable_default_handler()
        self.logger = self.context.logger

    def optimize(self):
        
        self.logger.info("Start optimization.")

        #search_space = {
        #    'num_conv_layers': [1, 2, 3, 4, 5],
        #    'num_dense_layers': [1, 2, 3, 4, 5],
        #    'conv_booster': [1, 2, 3, 4, 5]
        #}

        # Define the GridSampler with the search space
        #sampler = optuna.samplers.GridSampler(search_space)

        # Create a study object and use grid search
        #study = optuna.create_study(sampler=sampler, direction='maximize')
        
        #Create a study object and optimize the objective function
        study = optuna.create_study(direction='maximize')

        try:
            study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user.")
        #Logging of the best parameters
        self.logger.info(f'Best parameters: {study.best_params}')
        self.logger.info(f'Best value: {study.best_value}')

    #Define an objective function to be maximized.
    def objective(self, trial):

        #Suggest values of the hyperparameters using a trial object.
        #batch_size = trial.trial.suggest_categorical('batch_size', [32, 64])
        learning_rate = trial.suggest_float('lr', 1e-5, 1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1, log=True)
        #num_conv_layers =  trial.suggest_int('num_conv_layers', 1, 5)
        #num_dense_layers = trial.suggest_int('num_dense_layers', 1, 5)
        #conv_booster = trial.suggest_int('conv_booster', 1, 5)
        linear_decay = trial.suggest_float('linear_decay', 0, 2)

        self.oracle_config['parameters']['optimizer']['parameters']['lr'] = learning_rate
        self.oracle_config['parameters']['optimizer']['parameters']['weight_decay'] = weight_decay
        self.oracle_config['parameters']['model']['parameters']['num_conv_layers'] = 4 #num_conv_layers
        self.oracle_config['parameters']['model']['parameters']['num_dense_layers'] = 2 #num_dense_layers
        self.oracle_config['parameters']['model']['parameters']['conv_booster'] = 5 #conv_booster
        self.oracle_config['parameters']['model']['parameters']['linear_decay'] = linear_decay

        dataset = self.context.factories['datasets'].get_dataset(self.dataset_config)
        parameters = self.oracle_config['parameters']
        self.logger.info(f'Trial {trial.number}: Hyperparameters: {parameters}')
        oracle = self.context.factories['oracles'].get_oracle(self.oracle_config, dataset)
        
        # Check for accuracy metric
        if hasattr(oracle, 'mean_accuracy'):
            mean_accuracy = oracle.mean_accuracy
            if mean_accuracy is None:
                self.logger.error("The oracle did not return a valid accuracy value.")
            else:
                return mean_accuracy
        else:
            self.logger.error("Oracle does not have 'mean_accuracy' attribute.")
