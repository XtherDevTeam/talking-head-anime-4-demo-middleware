import tha4_api.animation as animation
import tha4_api.api
import pathlib
import PIL.Image
import json
import torch

manager = tha4_api.api.ImageInferenceManager(tha4_api.api.ImageInferenceManager.load_model(torch.device('cuda:0')), torch.device('cuda:0'))
manager.set_base_image(PIL.Image.open('data/character_models/lambda_01/character.png'))
params = tha4_api.api.PoseUpdateParams()
conf = animation.AnimationConfiguration(json.loads(pathlib.Path("./src/tha4_api/test_configuration.json").read_text()))
renderer = animation.Renderer(conf, manager, 20)
# print('\n\n'.join(i.__repr__() for i in renderer.compose_state('idle')))
renderer.compile_all_animations()
renderer.serailize('test_cache_elysia.tha4')
# renderer.deserialize('test_cache_elysia.tha4')
# renderer.render_animation(renderer.compose_state('idle'), debugging=True)