import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.networks.encoder import ImageEncoder

class PoseRNN(nn.Module):
  '''
  The backbone model. CNN + 2D LSTM.
  Given an input video, output the mean and standard deviation of the pose
  vectors (initial pose + beta) of each component.
  '''
  def __init__(self, n_components, n_frames_output, n_channels, image_size,
               image_latent_size, hidden_size, ngf, output_size, independent_components):
    super(PoseRNN, self).__init__()

    n_layers = int(np.log2(image_size)) - 1
    self.image_encoder = ImageEncoder(n_channels, image_latent_size, ngf, n_layers)
    # Encoder
    self.encode_rnn = nn.LSTM(image_latent_size + hidden_size, hidden_size,
                              num_layers=1, batch_first=True)
    if independent_components:
      predict_input_size = hidden_size
    else:
      predict_input_size = hidden_size * 2
    self.predict_rnn = nn.LSTM(predict_input_size, hidden_size, num_layers=1, batch_first=True)

    # Beta
    self.beta_mu_layer = nn.Linear(hidden_size, output_size)
    self.beta_sigma_layer = nn.Linear(hidden_size, output_size)

    # Initial pose
    self.initial_pose_rnn = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
    self.initial_pose_mu = nn.Linear(hidden_size, output_size)
    self.initial_pose_sigma = nn.Linear(hidden_size, output_size)

    self.n_components = n_components
    self.n_frames_output = n_frames_output
    self.image_latent_size = image_latent_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.independent_components = independent_components

  def get_initial_pose(self, repr):
    '''
    Get initial pose of each component.
    '''
    # Repeat first input representation.
    output, _ = self.initial_pose_rnn(repr)
    output = output.contiguous().view(-1, self.hidden_size)
    initial_mu = self.initial_pose_mu(output)
    initial_sigma = self.initial_pose_sigma(output)
    initial_sigma = F.softplus(initial_sigma)
    return initial_mu, initial_sigma

  def encode(self, input):
    '''
    First part of the model.
    input: video of size (batch_size, n_frames_input, n_channels, H, W)
    Return initial pose and input betas.
    '''
    batch_size, n_frames_input, n_channels, H, W = input.size()
    # encode each frame
    input_reprs = self.image_encoder(input.view(-1, n_channels, H, W))
    input_reprs = input_reprs.view(batch_size, n_frames_input, -1)
    # Initial zero hidden states (as input to lstm)
    prev_hidden = [Variable(torch.zeros(batch_size, 1, self.hidden_size).cuda())] * n_frames_input

    encoder_outputs = [] # all components
    hidden_states = []
    first_hidden_states = []
    for i in range(self.n_components):
      frame_outputs = []
      hidden = None
      for j in range(n_frames_input):
        rnn_input = torch.cat([input_reprs[:, j:(j+1), :], prev_hidden[j]], dim=2)
        output, hidden = self.encode_rnn(rnn_input, hidden)
        h = torch.cat([hidden[0][0:1], hidden[0][1:]], dim=2)
        c = torch.cat([hidden[1][0:1], hidden[1][1:]], dim=2)
        prev_hidden[j] = h.view(batch_size, 1, -1)
        if j == 0:
          # Save first hidden state
          first_hidden_states.append(h.view(batch_size, 1, -1))
        frame_outputs.append(output)

      # frame_outputs: batch_size x n_frames_output x hidden_size
      frame_outputs = torch.cat(frame_outputs, dim=1) # for 1 component
      encoder_outputs.append(frame_outputs)
      # Save last hidden states (h, c)
      hidden_states.append((h, c))

    # batch_size x n_frames_input x n_components x hidden_size
    encoder_outputs = torch.stack(encoder_outputs, dim=2)
    input_beta_mu = self.beta_mu_layer(encoder_outputs).view(-1, self.output_size)
    input_beta_sigma = self.beta_sigma_layer(encoder_outputs).view(-1, self.output_size)
    input_beta_sigma = F.softplus(input_beta_sigma)

    # Get initial pose
    first_hidden_states = torch.cat(first_hidden_states, dim=1)
    initial_mu, initial_sigma = self.get_initial_pose(first_hidden_states)
    return input_beta_mu, input_beta_sigma, initial_mu, initial_sigma,\
           encoder_outputs, hidden_states

  def predict(self, encoder_outputs, hidden_states):
    '''
    Second part of the model.
    input: encoder outputs and hidden_states of each component.
    Return predicted betas.
    '''
    batch_size = encoder_outputs.size(0)
    pred_beta_mu, pred_beta_sigma = None, None
    pred_outputs = []
    prev_hidden = [Variable(torch.zeros(batch_size, 1, self.hidden_size).cuda())] \
                       * self.n_frames_output
    for i in range(self.n_components):
      hidden = hidden_states[i]
      prev_outputs = encoder_outputs[:, -1:, i, :]
      frame_outputs = []
      # Manual unroll
      for j in range(self.n_frames_output):
        if self.independent_components:
          rnn_input = prev_outputs
        else:
          rnn_input = torch.cat([prev_outputs, prev_hidden[j]], dim=2)
        output, hidden = self.predict_rnn(rnn_input, hidden)
        prev_outputs = output
        prev_hidden[j] = hidden[0].view(batch_size, 1, -1)
        frame_outputs.append(output)
      # frame_outputs: batch_size x n_frames_output x (hidden_size * 2)
      frame_outputs = torch.cat(frame_outputs, dim=1)
      pred_outputs.append(frame_outputs)

    # batch_size x n_frames_output x n_components x hidden_size
    pred_outputs = torch.stack(pred_outputs, dim=2)
    pred_beta_mu = self.beta_mu_layer(pred_outputs).view(-1, self.output_size)
    pred_beta_sigma = self.beta_sigma_layer(pred_outputs).view(-1, self.output_size)
    pred_beta_sigma = F.softplus(pred_beta_sigma)
    return pred_beta_mu, pred_beta_sigma

  def forward(self, input):
    '''
    param input: video of size (batch_size, n_frames_input, n_channels, H, W)
    Output: input_beta: mean and std (for reconstruction), shape
                (batch_size, n_frames_input, n_components, output_size)
            pred_beta: mean and std (for prediction), shape
                (batch_size, n_frames_output, n_components, output_size)
            initial_pose: mean and std
    '''
    input_beta_mu, input_beta_sigma, initial_mu, initial_sigma,\
        encoder_outputs, hidden_states = self.encode(input)
    pred_beta_mu, pred_beta_sigma = self.predict(encoder_outputs, hidden_states)

    return input_beta_mu, input_beta_sigma, pred_beta_mu, pred_beta_sigma,\
           initial_mu, initial_sigma
