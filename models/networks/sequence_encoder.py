import torch
import torch.nn as nn

class SequenceEncoder(nn.Module):
  '''
  Encode a sequence of input vectors into a single vector.
  '''
  def __init__(self, input_size, hidden_size, output_size, num_layers=1):
    super(SequenceEncoder, self).__init__()

    self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
    self.out_layer = nn.Linear(hidden_size, output_size)

    self.input_size = input_size
    self.hidden_size = hidden_size

  def forward(self, input):
    '''
    input: temporal factor of size (batch_size, n_frames, input_size)
    output: prediction of size (batch_size, n_frames, output_size)
    '''
    encoder_output, hidden = self.encoder(input)
    last_output = encoder_output[:, -1, :]
    output = self.out_layer(last_output)
    return output
