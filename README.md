# JTK-Challenge

### Environment setup
Python virtual environments are used for this project. Execute the commands below in terminal to install all requirements.
```bash
~$chmod +x setup.sh
~$sh setup.sh
~$source jtk/bin/activate
```

### Preprocessing the dataset
Steps were taken to ensure new datasets can be easily generated for prediction durations. Major folder names or the prediction fail window can be modified in the environment file. To execute any of the scripts, the virtual environment `jtk` must be active.
1. Generate the labels for training the data. 
    ```bash
    ~$python scripts/preprocess_labels.py
    ```
    **Note**: This creates a folder **`./Processed_14`** where 14 is the specified SLIDE_N in the environment file. 
2. Remove local outliers and crop the data to daily windows for training and prediction
    ```bash
    ~$python scripts/crop_processed_data.py -c=True
    ```
    **Note**: This creates a folder **`./Cropped_14`**. The script is executed across multiple-processors because of the numerous *for* loops and it takes a while to run (roughly 1.5 hours on an old macbook with 4 processors).

Both scripts only need to be run once.


### Training the model
1. Arguments to the train script can be specified. By default, the data folder is obtained from the env file. Additional arguments like
    - `split` - train-test-split percentage.
        - validation data is default to 10% of training data
    - `batch_size` - batch size of the training dataset
    - `learning_rate` - to update the model weights
    - `num_epochs` - number of epochs to train the model for
    - `dropout` - percentage of dropout during training
    - `num_layers` - number of layers you want the model to have.
    ```bash
    ~$python scripts/train.py
    ```

### Inference


### To-Do
- [x] Complete data cleaning for spikes
- [x] Resample data for cropping
- [x] Crop timeseries for training model
- [x] Handle nan values
- [x] Create model architecture with LSTM
- [ ] Improve model architecture



You can specify a `SLIDE_N` value in the environment file to generate binary labels for number of days before an ESP test fails.




`num_hidden` is simply the dimension of the hidden state.<br>
The number of hidden layers is something else entirely. You can stack LSTMs on top of each other, 
so that the output of the first LSTM layer is the input to the second LSTM layer and so on. 
The number of hidden layers is how many LSTMs you stack on top of each other.