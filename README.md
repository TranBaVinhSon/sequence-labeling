# Sequence-labeling
Sequence Labeling implemented by Bi-LSTM using Tensorflow.

## Benchmark for building a (Bidirectional) LSTM model.

The hyperparameters used in the model:
- **learning_rate** - the initial value of the learning rate
- **max_lr_epoch** - after **max_lr_epoch**, the learning rate will be decreased
- **num_layers** - the number of (Bi)LSTM layers
- **num_steps** - the number of unrolled steps of (Bi)LSTM
- **hidden_size** - the number of (Bi)LSTM units
- **num_epochs** - the total number of epochs for training
- **keep_prob** - the probability of keeping weights in the dropout layer
- **lr_decay** - the decay of the learning rate
- **batch_size** - number of inputs

## Evaluation

- Accuracy
- F1 Score

## Todo

- [ ] BiLSTM + CRF
- [ ] BiLSTM + CRF + CNN
