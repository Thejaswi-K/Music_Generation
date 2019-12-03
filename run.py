import subprocess

# normal run
print('***starting normal run***')
subprocess.call(['python', 'train.py', '--data_dir=data/chopin-midi'])
subprocess.call(['python', 'sample.py', '--num_files=1'])
print('***normal run completed***')

# doubling the number of RNN neurons
rsubprocess.call(['python', 'train.py', '--data_dir=data/chopin-midi', '--rnn_size=128'])
subprocess.call(['python', 'sample.py', '--num_files=1'])
print('***128 RNN layers run completed***')

#experimenting with different optimizer, default is adam
print('***starting sgd run***')
subprocess.call(['python', 'train.py', '--data_dir=data/chopin-midi', '--optimizer=sgd'])
subprocess.call(['python', 'sample.py', '--num_files=1'])
print('***sgd run completed***')

print('***starting rmsprop run***')
subprocess.call(['python', 'train.py', '--data_dir=data/chopin-midi', '--optimizer=rmsprop'])
subprocess.call(['python', 'sample.py', '--num_files=1'])
print('***rmsprop run completed***')

print('***starting adagrad run***')
subprocess.call(['python', 'train.py', '--data_dir=data/chopin-midi', '--optimizer=adagrad'])
subprocess.call(['python', 'sample.py', '--num_files=1'])
print('***adagrad run completed***')

print('***starting adadelta run***')
subprocess.call(['python', 'train.py', '--data_dir=data/chopin-midi', '--optimizer=adadelta'])
subprocess.call(['python', 'sample.py', '--num_files=1'])
print('***adadelta run completed***')

print('***starting adamax run***')
subprocess.call(['python', 'train.py', '--data_dir=data/chopin-midi', '--optimizer=adamax'])
subprocess.call(['python', 'sample.py', '--num_files=1'])
print('***adamax run completed***')

print('***starting nadam run***')
subprocess.call(['python', 'train.py', '--data_dir=data/chopin-midi', '--optimizer=nadam'])
subprocess.call(['python', 'sample.py', '--num_files=1'])
print('***nadam run completed***')

# dropout increase
print('***starting dropout increase run***')
subprocess.call(['python', 'train.py', '--data_dir=data/chopin-midi', '--dropout=0.5'])
subprocess.call(['python', 'sample.py', '--num_files=1'])
print('***dropout increase run completed***')

# increasing the number of layers to three layers,
# lower the number of neurons to 32 neurons,
# lowering the batch size to 1,
# increasing the number of epochs to 20 epochs
print('***starting dropout increase run***')
subprocess.call(['python', 'train.py', '--data_dir=data/chopin-midi', '--num_layers=3', '--rnn_size=32', '--batch_size=1', '--num_epochs=20', '--n_jobs=4'])
subprocess.call(['python', 'sample.py', '--num_files=1'])
print('***dropout increase run completed***')