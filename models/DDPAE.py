from collections import defaultdict
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO

from .base_model import BaseModel
from models.networks.pose_rnn import PoseRNN
from models.networks.sequence_encoder import SequenceEncoder
from models.networks.encoder import ImageEncoder
from models.networks.decoder import ImageDecoder
import utils


class DDPAE(BaseModel):
  '''
  The DDPAE model.
  '''
  def __init__(self, opt):
    super(DDPAE, self).__init__()

    self.is_train = opt.is_train
    assert opt.image_size[0] == opt.image_size[1]
    self.image_size = opt.image_size[0]
    print('Image size: {}'.format(self.image_size))
    self.object_size = self.image_size // 2

    # Data parameters
    # self.__dict__.update(opt.__dict__)
    self.n_channels = opt.n_channels
    self.n_components = opt.n_components
    self.total_components = self.n_components
    self.batch_size = opt.batch_size
    self.n_frames_input = opt.n_frames_input
    self.n_frames_output = opt.n_frames_output
    self.n_frames_total = self.n_frames_input + self.n_frames_output
    # Hyperparameters
    self.image_latent_size = opt.image_latent_size
    self.content_latent_size = opt.content_latent_size
    self.pose_latent_size = opt.pose_latent_size
    self.hidden_size = opt.hidden_size
    self.ngf = opt.ngf
    self.independent_components = opt.independent_components
    self.predict_loss_only = False

    # Training parameters
    if opt.is_train:
      self.lr_init = opt.lr_init
      self.lr_decay = opt.lr_decay
      self.when_to_predict_only = opt.when_to_predict_only

    # Networks
    self.setup_networks()

    # Priors
    self.scale = opt.stn_scale_prior
    # Initial pose prior
    self.initial_pose_prior_mu = Variable(torch.cuda.FloatTensor([self.scale, 0, 0]))
    self.initial_pose_prior_sigma = Variable(torch.cuda.FloatTensor([0.2, 1, 1]))
    # Beta prior
    sd = 0.1
    self.beta_prior_mu = Variable(torch.zeros(self.pose_latent_size).cuda())
    self.beta_prior_sigma = Variable(torch.ones(self.pose_latent_size).cuda() * sd)

  def setup_networks(self):
    '''
    Networks for DDPAE.
    '''
    self.nets = {}
    # These will be registered in model() and guide() with pyro.module().
    self.model_modules = {}
    self.guide_modules = {}

    # Backbone, Pose RNN
    pose_model = PoseRNN(self.n_components, self.n_frames_output, self.n_channels,
                         self.image_size, self.image_latent_size, self.hidden_size,
                         self.ngf, self.pose_latent_size, self.independent_components)
    self.pose_model = nn.DataParallel(pose_model.cuda())

    self.nets['pose_model'] = self.pose_model
    self.guide_modules['pose_model'] = self.pose_model

    # Content LSTM
    content_lstm = SequenceEncoder(self.content_latent_size, self.hidden_size,
                                   self.content_latent_size * 2)
    self.content_lstm = nn.DataParallel(content_lstm.cuda())
    self.nets['content_lstm'] = self.content_lstm
    self.model_modules['content_lstm'] = self.content_lstm

    # Image encoder and decoder
    n_layers = int(np.log2(self.object_size)) - 1
    object_encoder = ImageEncoder(self.n_channels, self.content_latent_size,
                                  self.ngf, n_layers)
    object_decoder = ImageDecoder(self.content_latent_size, self.n_channels,
                                  self.ngf, n_layers, 'sigmoid')
    self.object_encoder = nn.DataParallel(object_encoder.cuda())
    self.object_decoder = nn.DataParallel(object_decoder.cuda())
    self.nets.update({'object_encoder': self.object_encoder,
                      'object_decoder': self.object_decoder})
    self.model_modules['decoder'] = self.object_decoder
    self.guide_modules['encoder'] = self.object_encoder

  def setup_training(self):
    '''
    Setup Pyro SVI, optimizers.
    '''
    if not self.is_train:
      return

    self.pyro_optimizer = optim.Adam({'lr': self.lr_init})
    self.svis = {'elbo': SVI(self.model, self.guide, self.pyro_optimizer, loss=Trace_ELBO())}

    # Separate pose_model parameters and other networks' parameters
    params = []
    for name, net in self.nets.items():
      if name != 'pose_model':
        params.append(net.parameters())
    self.optimizer = torch.optim.Adam(\
                     [{'params': self.pose_model.parameters(), 'lr': self.lr_init},
                      {'params': itertools.chain(*params), 'lr': self.lr_init}
                     ], betas=(0.5, 0.999))

  def get_objects(self, input, transformer):
    '''
    Crop objects from input given the transformer.
    '''
    # Repeat input: batch_size x n_frames_input x n_components x C x H x W
    repeated_input = torch.stack([input] * self.n_components, dim=2)
    repeated_input = repeated_input.view(-1, *input.size()[-3:])
    # Crop objects
    transformer = transformer.contiguous().view(-1, transformer.size(-1))
    input_obj = utils.image_to_object(repeated_input, transformer, self.object_size)
    input_obj = input_obj.view(-1, *input_obj.size()[-3:])
    return input_obj

  def constrain_pose(self, pose):
    '''
    Constrain the value of the pose vectors.
    '''
    # Makes training faster.
    scale = torch.clamp(pose[..., :1], self.scale - 1, self.scale + 1)
    xy = F.tanh(pose[..., 1:]) * (scale - 0.5)
    pose = torch.cat([scale, xy], dim=-1)
    return pose

  def sample_latent(self, input, input_latent_mu, input_latent_sigma, pred_latent_mu,
                    pred_latent_sigma, initial_pose_mu, initial_pose_sigma, sample=True):
    '''
    Return latent variables: dictionary containing pose and content.
    Then, crop objects from the images and encode into z.
    '''
    latent = defaultdict(lambda: None)

    beta = self.get_transitions(input_latent_mu, input_latent_sigma,
                                pred_latent_mu, pred_latent_sigma, sample)
    pose = self.accumulate_pose(beta)
    # Sample initial pose
    initial_pose = self.pyro_sample('initial_pose', dist.Normal, initial_pose_mu,
                                    initial_pose_sigma, sample)
    pose += initial_pose.view(-1, 1, self.n_components, self.pose_latent_size)
    pose = self.constrain_pose(pose)

    # Get input objects
    input_pose = pose[:, :self.n_frames_input, :, :]
    input_obj = self.get_objects(input, input_pose)
    # Encode the sampled objects
    z = self.object_encoder(input_obj)
    z = self.sample_content(z, sample)
    latent.update({'pose': pose, 'content': z})
    return latent

  def sample_latent_prior(self, input):
    '''
    Return latent variables: [pose, z], sampled from prior distribution.
    '''
    latent = defaultdict(lambda: None)

    batch_size = input.size(0)
    # z prior
    N = batch_size * self.total_components
    z_prior_mu = Variable(torch.zeros(N, self.content_latent_size).cuda())
    z_prior_sigma = Variable(torch.ones(N, self.content_latent_size).cuda())
    z = self.pyro_sample('content', dist.Normal, z_prior_mu, z_prior_sigma, sample=True)

    # input_beta prior
    N = batch_size * self.n_frames_input * self.n_components
    input_beta_prior_mu = self.beta_prior_mu.repeat(N, 1)
    input_beta_prior_sigma = self.beta_prior_sigma.repeat(N, 1)
    input_beta = self.pyro_sample('input_beta', dist.Normal, input_beta_prior_mu,
                                  input_beta_prior_sigma, sample=True)
    beta = input_beta.view(batch_size, self.n_frames_input, self.n_components,
                           self.pose_latent_size)

    # pred_beta prior
    M = batch_size * self.n_frames_output * self.n_components
    pred_beta_prior_mu = self.beta_prior_mu.repeat(M, 1)
    pred_beta_prior_sigma = self.beta_prior_sigma.repeat(M, 1)
    pred_beta = self.pyro_sample('pred_beta', dist.Normal, pred_beta_prior_mu,
                                 pred_beta_prior_sigma, sample=True)
    pred_beta = pred_beta.view(batch_size, self.n_frames_output, self.n_components,
                               self.pose_latent_size)
    beta = torch.cat([beta, pred_beta], dim=1)

    # Get pose
    pose = self.accumulate_pose(beta)
    # Sample and add initial pose
    N = batch_size * self.n_components
    initial_pose_prior_mu = self.initial_pose_prior_mu.repeat(N, 1)
    initial_pose_prior_sigma = self.initial_pose_prior_sigma.repeat(N, 1)
    initial_pose = self.pyro_sample('initial_pose', dist.Normal, initial_pose_prior_mu,
                                    initial_pose_prior_sigma, sample=True)
    pose += initial_pose.view(-1, 1, self.n_components, self.pose_latent_size)
    pose = self.constrain_pose(pose)

    latent.update({'pose': pose, 'content': z})
    return latent

  def decode_components(self, latent):
    '''
    param latent: return value from self.sample_latent()
    Return values:
    components: (batch_size * n_frames * n_components) x n_channels x image_size x image_size
    '''
    pose, z = latent['pose'], latent['content']

    # (batch_size * n_frames_total * n_components) x content_latent_size
    z = z.view(-1, self.content_latent_size)
    objects = self.object_decoder(z)
    objects = objects.view(-1, *objects.size()[-3:]) # N x C x H x W
    pose = pose.view(-1, self.pose_latent_size)
    components = utils.object_to_image(objects, pose, self.image_size)
    latent['Y'] = objects
    return components

  def get_transitions(self, input_latent_mu, input_latent_sigma, pred_latent_mu,
                      pred_latent_sigma, sample=True):
    '''
    Sample the transition variables beta.
    '''
    # input_beta: (batch_size * n_frames_input * n_components) x pose_latent_size
    input_beta = self.pyro_sample('input_beta', dist.Normal, input_latent_mu,
                                  input_latent_sigma, sample)
    beta = input_beta.view(-1, self.n_frames_input, self.n_components, self.pose_latent_size)

    # pred_beta: (batch_size * n_frames_output) x n_components x pose_latent_size
    pred_beta = self.pyro_sample('pred_beta', dist.Normal, pred_latent_mu,
                                 pred_latent_sigma, sample)
    pred_beta = pred_beta.view(-1, self.n_frames_output, self.n_components,
                               self.pose_latent_size)
    # Concatenate the input and prediction beta
    beta = torch.cat([beta, pred_beta], dim=1)
    return beta

  def accumulate_pose(self, beta):
    '''
    Accumulate pose from the transition variables beta.
    pose_k = sum_{i=1}^k beta_k
    '''
    batch_size, n_frames, _, pose_latent_size = beta.size()
    accumulated = []
    for i in range(n_frames):
      if i == 0:
        p_i = beta[:, 0:1, :, :]
      else:
        p_i = beta[:, i:(i+1), :, :] + accumulated[-1]
      accumulated.append(p_i)
    accumulated = torch.cat(accumulated, dim=1)
    return accumulated

  def sample_content(self, content, sample):
    '''
    Pass into content_lstm to get a final content.
    '''
    content = content.view(-1, self.n_frames_input, self.total_components, self.content_latent_size)
    contents = []
    for i in range(self.total_components):
      z = content[:, :, i, :]
      z = self.content_lstm(z).unsqueeze(1) # batch_size x 1 x (content_latent_size * 2)
      contents.append(z)
    content = torch.cat(contents, dim=1).view(-1, self.content_latent_size * 2)

    # Get mu and sigma, and sample.
    content_mu = content[:, :self.content_latent_size]
    content_sigma = F.softplus(content[:, self.content_latent_size:])
    content = self.pyro_sample('content', dist.Normal, content_mu, content_sigma, sample)
    return content

  def get_output(self, components, latent):
    '''
    Take the sum of the components.
    '''
    # components: batch_size x n_frames_total x total_components x C x H x W
    batch_size = components.size(0)
    # Sum the components
    output = torch.sum(components, dim=2)
    output = torch.clamp(output, max=1)
    return output

  def encode(self, input, sample=True):
    '''
    Encode video with pose_model, and sample the latent variables for reconstruction
    and prediction.
    Note: pyro.sample is called in self.sample_latent().
    param input: video of size (batch_size, n_frames_input, C, H, W)
    param sample: True if this is called by guide(), and sample with pyro.sample.
    Return latent: a dictionary {'pose': pose, 'content': content, ...}
    '''
    input_latent_mu, input_latent_sigma, pred_latent_mu, pred_latent_sigma,\
        initial_pose_mu, initial_pose_sigma = self.pose_model(input)

    # Sample latent variables
    latent = self.sample_latent(input, input_latent_mu, input_latent_sigma, pred_latent_mu,
                                pred_latent_sigma, initial_pose_mu, initial_pose_sigma, sample)
    return latent

  def decode(self, latent, batch_size):
    '''
    Decode the latent variables into components, and produce the final output.
    param latent: dictionary, return values from self.encode()
    Return values:
    output: batch_size x n_frames_total x n_channels x image_size x image_size
    components: batch_size x n_frames_total x total_components x n_channels x image_size x image_size
    '''
    # Get the final content: batch_size x 1 x total_components x content_latent_size
    content = latent['content']
    content = content.view(batch_size, 1, self.total_components, content.size(-1))
    # Repeat the contents
    content = content.repeat(1, self.n_frames_total, 1, 1)
    latent['content'] = content

    components = self.decode_components(latent)
    components = components.view(-1, self.n_frames_total, self.total_components,
                                 self.n_channels, self.image_size, self.image_size)
    output = self.get_output(components, latent)
    return output, components

  def model(self, input, output):
    '''
    Likelihood model: sample from prior, then decode to video.
    param input: video of size (batch_size, self.n_frames_input, C, H, W)
    param output: video of size (batch_size, self.n_frames_output, C, H, W)
    '''
    # Register networks
    for name, net in self.model_modules.items():
      pyro.module(name, net)

    observation = torch.cat([input, output], dim=1)

    # Sample from prior
    latent = self.sample_latent_prior(input)
    # Decode
    decoded_output, components = self.decode(latent, input.size(0))
    decoded_output = decoded_output.view(*observation.size())
    if self.predict_loss_only:
      # Only consider loss from the predicted frames
      decoded_output = decoded_output[:, self.n_frames_input:]
      observation = observation[:, self.n_frames_input:]
      components = components[:, self.n_frames_input:, ...]

    # pyro observe
    sd = Variable(0.3 * torch.ones(*decoded_output.size()).cuda())
    pyro.sample('obs', dist.Normal(decoded_output, sd), obs=observation)

  def guide(self, input, output):
    '''
    Posterior model: encode input
    param input: video of size (batch_size, n_frames_input, C, H, W).
    parma output: not used.
    '''
    # Register networks
    for name, net in self.guide_modules.items():
      pyro.module(name, net)

    self.encode(input, sample=True)

  def train(self, input, output):
    '''
    param input: video of size (batch_size, n_frames_input, C, H, W)
    param output: video of size (batch_size, self.n_frames_output, C, H, W)
    Return video_dict, loss_dict
    '''
    input = Variable(input.cuda(), requires_grad=False)
    output = Variable(output.cuda(), requires_grad=False)
    assert input.size(1) == self.n_frames_input

    # SVI
    batch_size, _, C, H, W = input.size()
    numel = batch_size * self.n_frames_total * C * H * W
    loss_dict = {}
    for name, svi in self.svis.items():
      # loss = svi.step(input, output)
      # Note: backward() is already called in loss_and_grads.
      loss = svi.loss_and_grads(svi.model, svi.guide, input, output)
      loss_dict[name] = loss / numel

    # Update parameters
    self.optimizer.step()
    self.optimizer.zero_grad()

    return {}, loss_dict

  def test(self, input, output):
    '''
    Return decoded output.
    '''
    input = Variable(input.cuda())
    batch_size, _, _, H, W = input.size()
    output = Variable(output.cuda())
    gt = torch.cat([input, output], dim=1)

    latent = self.encode(input, sample=False)
    decoded_output, components = self.decode(latent, input.size(0))
    decoded_output = decoded_output.view(*gt.size())
    components = components.view(batch_size, self.n_frames_total, self.total_components,
                                 self.n_channels, H, W)
    latent['components'] = components
    decoded_output = decoded_output.clamp(0, 1)

    self.save_visuals(gt, decoded_output, components, latent)
    return decoded_output.cpu(), latent

  def save_visuals(self, gt, output, components, latent):
    '''
    Save results. Draw bounding boxes on each component.
    '''
    pose = latent['pose']
    components = components.detach().cpu()
    for i in range(self.n_components):
      p = pose.data[0, :, i, :].cpu()
      images = components.data[0, :, i, ...]
      images = utils.draw_components(images, p)
      components.data[0, :, i, ...] = images

    super(DDPAE, self).save_visuals(gt, output, components, latent)

  def update_hyperparameters(self, epoch, n_epochs):
    '''
    If when_to_predict_only > 0 and it halfway through training, then only train with
    prediction loss.
    '''
    lr_dict = super(DDPAE, self).update_hyperparameters(epoch, n_epochs)

    if self.when_to_predict_only > 0 and epoch > int(n_epochs * self.when_to_predict_only):
      self.predict_loss_only = True

    return lr_dict
