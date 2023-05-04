class SweepHandler():
    """
    This class is used to define the sweep parameters and the metric to be used for the sweep.
    """
    
    def __init__(self):

        self.metric = {
        'name': 'accuracy',
        'goal': 'maximize'   
        }

        self.parameters_dict = {
        'num_hidden': {
            'values': [1, 2, 3, 4]
            },
        'hidden_size': {
            'values': [32, 64, 128, 256, 512]
            },
        'dropout': {
                'values': [0.3, 0.4, 0.5]
            },
        'learning_rate': {
            'values': [0.001, 0.0005, 0.0001]
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
        