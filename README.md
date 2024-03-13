# JTK-Challenge

### Environment setup
Python virtual environments are used for this project. Execute the commands below in terminal to install all requirements.
```bash
~$chmod +x setup.sh
~$sh setup.sh
~$source jtk/bin/activate
```

### Preprocessing the dataset
First step of preprocessing is to generate the labels for training the data. You can specify a `SLIDE_N` value in the environment file to generate binary labels for number of days before an ESP test fails. <br>
To generate the labels, the virtual environment `jtk` must be active, then you execute the command below
```bash
~$python scripts/preprocess_labels.py
```
This creates a folder **`./Processed_14`** where 14 is the specified SLIDE_N in the environment file. This way, new datasets can be easily generated for different models.