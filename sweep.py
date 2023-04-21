class SweepHandler():
    def __init__(self):

        self.metric = {
        'name': 'accuracy',
        'goal': 'maximize'   
        }

        self.parameters_dict = {
        'optimizer': {
            'values': ['adam', 'adamW']
            },
        'num_hidden': {
            'values': [1, 2, 3, 4, 5]
            },
        'hidden_size': {
            'values': [32, 64, 128, 256, 512]
            },
        'dropout': {
                'values': [0.1, 0.3, 0.4, 0.5]
            },
        'learning_rate': {
            'values': [0.1, 0.01, 0.001, 0.0001]
        },

        'NeuronType': {
            'values': ['Leaky', 'Synaptic']
            },
        'learn_alpha': {
            'values': [True, False]
            },
        'learn_beta': { 
            'values': [True, False]
            },
        'learn_threshold': {
            'values': [True, False]
            },
        'surrogate_gradient': {
            'values': ["atan", "sigmoid", "fast_sigmoid"]
            },
        
            
        }
        """'features': {
            'values': ["pure", "pure_gradient", "gradient", "hf_pure", "hf_gradient", "all"]
            },"""
    """def get_params(self, config):
        self.hidden_size = [config.hidden_size]*config.num_hidden
      
        if config.surrogate_gradient == "atan":
            spike_grad = surrogate.atan()

        elif config.surrogate_gradient == "sigmoid":
            spike_grad = surrogate.sigmoid()
        
        elif config.surrogate_gradient == "fast_sigmoid":
            spike_grad = surrogate.fast_sigmoid()

        # replace previously defined network with the one initialized by wandb sweep
        self.network = SNN(input_size=input_size, hidden_size=hidden_size, dropout=config.dropout, neuron_type=config.NeuronType, learn_alpha=config.learn_alpha, learn_beta=config.learn_beta, learn_threshold=config.learn_threshold, spike_grad=spike_grad).to(opt.device)

        if config.optimizer == "adam":
            optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
        elif config.optimizer == "adamW":
            optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate)"""