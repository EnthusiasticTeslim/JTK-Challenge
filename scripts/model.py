import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size=32, n_layer=1, num_stack_layers=1, dropout=0.75) -> None:
        """
        LSTM Classifier model

        Args:
            n_features (int): 13 features available from the ESP data
            n_classes (int): defaults to 1 because we're doing binary classification and we can scale pribabilities.
            hidden_size (int, optional): hidden size of the LSTM. Defaults to 32.
            num_stack_layers (int, optional): Number of LSTMS stacked on each other. Defaults to 1.
            n_layer (int, optional): _description_. Defaults to 3.
            dropout (float, optional): _description_. Defaults to 0.75.
        """
        super(LSTMClassifier, self).__init__()

        self.lstm_layers = nn.ModuleList()
        self.num_stack_layers = num_stack_layers

        dropout = dropout if n_layer > 1 else 0
            
        current_input_dim = n_features
        for i in range(self.num_stack_layers):
            # Calculate the hidden size for the current layer
            layer_hidden_size = hidden_size * (2 ** i) if i < self.num_stack_layers // 2 else hidden_size * (2 ** (self.num_stack_layers - i - 1))

            self.lstm_layers.append(nn.LSTM(current_input_dim, layer_hidden_size, batch_first=True,
                                            num_layers=n_layer, dropout=dropout))
            
            # The next layer's input dimension is the current layer's output dimension
            current_input_dim = layer_hidden_size

        
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for lstm_layer in self.lstm_layers:
            lstm_layer.flatten_parameters()
            x, (hidden, _) = lstm_layer(x)
                
        # Extract the last state of the last layer
        out = hidden[-1]
        out = self.batch_norm(out)
        out = self.classifier(out)

        return self.sigmoid(out)