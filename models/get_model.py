from .DDPAE import DDPAE

def get_model(opt):
  if opt.model == 'crop':
    model = DDPAE(opt)
  else:
    raise NotImplementedError

  model.setup_training()
  model.initialize_weights()
  return model
