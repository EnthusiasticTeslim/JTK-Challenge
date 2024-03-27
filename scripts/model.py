import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=32, n_layer=1, dropout=0.75) -> None:
        """
        LSTM Classifier model

        Args:
            n_features (int): 13 features available from the ESP data
            n_classes (int): defaults to 1 because we're doing binary classification and we can scale pribabilities.
            n_hidden (int, optional): Number of LSTMS stacked on each other. Defaults to 32.
            n_layer (int, optional): _description_. Defaults to 3.
            dropout (float, optional): _description_. Defaults to 0.75.
        """
        super(LSTMClassifier, self).__init__()

        self.lstm1 = nn.LSTM(input_size=n_features,
                             hidden_size=n_hidden*2,
                             num_layers=n_layer, 
                             batch_first=True, 
                             dropout=dropout,)
        
        self.classifier = nn.Linear(n_hidden*2, n_classes)

    def forward(self, x):
        self.lstm1.flatten_parameters()
        _, (hidden, _) = self.lstm1(x)

        # Extract the last state of the last layer
        out = hidden[-1]
        return self.classifier(out)