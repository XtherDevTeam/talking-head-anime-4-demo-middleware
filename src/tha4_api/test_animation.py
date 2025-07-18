import tha4_api.animation as animation
import tha4_api.api
import pathlib
import PIL.Image
import json
import torch

manager = tha4_api.api.ImageInferenceManager(tha4_api.api.ImageInferenceManager.load_model(torch.device('cuda:0')), torch.device('cuda:0'))
manager.set_base_image(PIL.Image.open('data/character_models/lambda_01/Layer 1_With_Mouth.png'))
params = tha4_api.api.PoseUpdateParams()
conf = animation.AnimationConfiguration(json.loads(pathlib.Path("data/character_models/lambda_01/test_configuration.json").read_text()))
renderer = animation.Renderer(conf, manager, 20)
# print('\n\n'.join(i.__repr__() for i in renderer.compose_state('idle')))
# renderer.deserialize('tha4_cache/Cyrene.tha4')
# renderer.configuration = conf
renderer.compile_all_animations()
renderer.serailize('tha4_cache/Cyrene.tha4')
# renderer.render_animation(renderer.compose_state('idle'), debugging=True)