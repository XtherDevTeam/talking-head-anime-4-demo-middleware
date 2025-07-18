import tha4_api.animation
import tha4_api.api
import livekit.api
import livekit.rtc
import typing
import pathlib
import PIL.Image
import torch
import io
import asyncio
import json
import numpy
import queue
import time
import uuid
import threading
from tha4_api import logger
import cv2


manager = tha4_api.api.ImageInferenceManager(tha4_api.api.ImageInferenceManager.load_model(torch.device('cuda:0')), torch.device('cuda:0'))
manager.set_base_image(PIL.Image.open('data/character_models/lambda_01/character.png'))
params = tha4_api.api.PoseUpdateParams()
conf = tha4_api.animation.AnimationConfiguration(json.loads(pathlib.Path("./src/tha4_api/test_configuration_edited.json").read_text()))
renderer = tha4_api.animation.Renderer(conf, manager, 20)
# print('\n\n'.join(i.__repr__() for i in renderer.compose_state('idle')))
# renderer.compile_all_animations()
# renderer.serailize('test_cache_elysia.tha4')
renderer.deserialize('tha4_cache/Cyrene.tha4')

for i in renderer.configuration.states:
    logger.Logger.log(f"Composing state {i}")
    all = []
    for j in renderer.configuration.states[i]:
        all += renderer.compose_animation_group(j)
        
    renderer.render_animation(all, debugging=True)
