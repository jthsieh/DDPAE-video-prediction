import glob
import numpy as np
import os
from PIL import Image

import data
import models
import utils
from utils.visualizer import Visualizer

def save_images(prediction, gt, latent, save_dir, step):
  pose, components = latent['pose'].data.cpu(), latent['components'].data.cpu()
  batch_size, n_frames_total = prediction.shape[:2]
  n_components = components.shape[2]
  for i in range(batch_size):
    filename = '{:05d}.png'.format(step)
    y = gt[i, ...]
    rows = [y]
    if n_components > 1:
      for j in range(n_components):
        p = pose[i, :, j, :]
        comp = components[i, :, j, ...]
        if pose.size(-1) == 3:
          comp = utils.draw_components(comp, p)
        rows.append(utils.to_numpy(comp))
    x = prediction[i, ...]
    rows.append(x)
    # Make a grid of 4 x n_frames_total images
    image = np.concatenate(rows, axis=2).squeeze(1)
    image = np.concatenate([image[i] for i in range(n_frames_total)], axis=1)
    image = (image * 255).astype(np.uint8)
    # Save image
    Image.fromarray(image).save(os.path.join(save_dir, filename))
    step += 1

  return step

def evaluate(opt, dloader, model, use_saved_file=False):
  # Visualizer
  if hasattr(opt, 'save_visuals') and opt.save_visuals:
    vis = Visualizer(os.path.join(opt.ckpt_path, 'tb_test'))
  else:
    opt.save_visuals = False

  model.setup(is_train=False)
  metric = utils.Metrics()
  results = {}

  if hasattr(opt, 'save_all_results') and opt.save_all_results:
    save_dir = os.path.join(opt.ckpt_path, 'results')
    os.makedirs(save_dir, exist_ok=True)
  else:
    opt.save_all_results = False

  # Hacky
  is_bouncing_balls = ('bouncing_balls' in opt.dset_name) and opt.n_components == 4
  if is_bouncing_balls:
    dloader.dataset.return_positions = True
    saved_positions = os.path.join(opt.ckpt_path, 'positions.npy') if use_saved_file else ''
    velocity_metric = utils.VelocityMetrics(saved_positions)

  count = 0
  for step, data in enumerate(dloader):
    if not is_bouncing_balls:
      input, gt = data
    else:
      input, gt, positions = data
    output, latent = model.test(input, gt)
    pred = output[:, opt.n_frames_input:, ...]
    metric.update(gt, pred)

    if opt.save_all_results:
      gt = np.concatenate([input.numpy(), gt.numpy()], axis=1)
      prediction = utils.to_numpy(output)
      count = save_images(prediction, gt, latent, save_dir, count)

    if is_bouncing_balls:
      # Calculate position and velocity from pose
      pose = latent['pose'].data.cpu()
      velocity_metric.update(positions, pose, opt.n_frames_input)

    if (step + 1) % opt.log_every == 0:
      print('{}/{}'.format(step + 1, len(dloader)))
      if opt.save_visuals:
        vis.add_images(model.get_visuals(), step, prefix='test')

  # BCE, MSE
  results.update(metric.get_scores())

  if is_bouncing_balls:
    # Don't break the original code
    dloader.dataset.return_positions = False
    results.update(velocity_metric.get_scores())

  return results

def main():
  opt, logger, vis = utils.build(is_train=False)

  dloader = data.get_data_loader(opt)
  print('Val dataset: {}'.format(len(dloader.dataset)))
  model = models.get_model(opt)

  for epoch in opt.which_epochs:
    # Load checkpoint
    if epoch == -1:
      # Find the latest checkpoint
      checkpoints = glob.glob(os.path.join(opt.ckpt_path, 'net*.pth'))
      assert len(checkpoints) > 0
      epochs = [int(filename.split('_')[-1][:-4]) for filename in checkpoints]
      epoch = max(epochs)
    logger.print('Loading checkpoints from {}, epoch {}'.format(opt.ckpt_path, epoch))
    model.load(opt.ckpt_path, epoch)

    results = evaluate(opt, dloader, model)
    for metric in results:
      logger.print('{}: {}'.format(metric, results[metric]))

if __name__ == '__main__':
  main()
