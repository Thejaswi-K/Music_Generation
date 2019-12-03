# cmpe257-team-project

## Machine Learning Team Project - Music Generation

Install dependencies.

`pip install -r requirements.txt`

To generate MIDI files, run the 'run.py' script.

`python run.py`

This script will run all of the different experiments with the Keras LSTM model. It will then generate a MIDI file based on the model tweaks listed below.

Experiments:
1. Normal run (64 neurons, 10 epochs, 32 batch size, Adam optimizer)
2. Double the RNN neurons (64 -> 128)
3. SGD Optimizer
4. RMSPROP Optimizer
5. ADAGRAD Optimizer
6. ADADELTA Optimizer
7. ADAMAX Optimizer
8. NADAM Optimizer
9. Dropout increase (0.2 -> 0.5)
10. Increase number of layers to 3, decrease number of neurons to 32, increase number of epochs to 20.

To change the the LSTM model parameters for training, run:

`python train.py --help`

To change the which model to use for music generation or how many midi files get generated, run

`python sample.py --help`

The music generated will be in the 'experiments/xx/generated/' directory.
The tensorboard event log will be in the 'experiments/xx/tensorboard-logs' directory.

You can view the training metrics of the experiments by running:

`tensorboard --logdir experiments/`
