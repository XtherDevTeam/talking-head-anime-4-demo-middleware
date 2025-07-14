import tha4_api.api
import torch
import PIL.Image

manager = tha4_api.api.ImageInferenceManager(tha4_api.api.ImageInferenceManager.load_model(torch.device('cuda:0')), torch.device('cuda:0'))

manager.set_base_image(PIL.Image.open('data/character_models/lambda_01/character.png'))

params = tha4_api.api.PoseUpdateParams()

params.set_eyebrow_params('troubled', 1, 1)
params.set_head_rotation(0, 0, 0)

nd_image = manager.inference(params, 0)

# export nd_image to png
PIL.Image.fromarray(nd_image).save('output.png')
