import uuid
from tha4_api import logger
import tha4_api.api
import sys
import typing
import random
import queue
import threading
import time
import math
import numpy
import pickle
import pathlib
import gzip
import tha4_api.animationProvider
from moviepy import ImageSequenceClip
import io
import tempfile

class AnimationConfiguration:
    def __init__(self, configuration: dict):
        self.raw = configuration
        self.name: str = configuration.get("name", "Unnamed Animation Configuration")
        self.description: str = configuration.get("description", "No description provided.")
        self.states: dict[str, dict] = configuration.get("states", {})
        self.breathing: dict[str, dict] = configuration.get("breathing", {})
        
class RendererStateCacher:
    def __init__(self, renderer: 'Renderer'):
        self.renderer = renderer
        self.state_cache = {}
        
    def store_state(self, state: tha4_api.api.PoseUpdateParams, img: numpy.ndarray):
        self.state_cache[state] = img
        
    def get_state(self, state: tha4_api.api.PoseUpdateParams):
        if state in self.state_cache:
            return self.state_cache[state]
        else:
            img = self.renderer.imgInferenceManager.inference(state, 0)
            self.store_state(state, img)
            return img
        
    def serailize(self) -> bytes:
        data = pickle.dumps(self.state_cache)
        return gzip.compress(data)
    
    def deserialize(self, data: bytes):
        self.state_cache = pickle.loads(gzip.decompress(data))
    
        
class Renderer:
    def __init__(self, configuration: AnimationConfiguration, imgInferenceManager: tha4_api.api.ImageInferenceManager, baseFps: int = 20):
        self.configuration = configuration
        self.imgInferenceManager = imgInferenceManager
        self.defaultParams = tha4_api.api.PoseUpdateParams()
        self.baseFps = baseFps
        self.pending_states = queue.Queue()
        self.available_events = {
            'frame_update': []
        }
        self.last_breathing_state: typing.Optional[float] = None
        self.pending_breathing_state: typing.List[float] = []
        self.rendererStateCacher = RendererStateCacher(self)
        self.connected = False
        
    def switch_state(self, state: str) -> None:
        logger.Logger.log(f"Switching to state {state}")
        self.pending_states.put(state)
        
    def compile_all_animations(self):
        composed = []
        print("Composing animations...")
        for state_name, state_config in self.configuration.states.items():
            for sub_state in state_config:
                composed += self.compose_animation_group(sub_state)
        print("Done composing animations. Rendering...")
        begin = time.time()
        self.render_animation(composed)
        print(f"Done rendering in {time.time() - begin} seconds.")
        
    def serailize(self, dest: str):
        pathlib.Path(dest).write_bytes(pickle.dumps({
            'renderer_state_cacher': self.rendererStateCacher.serailize(),
            'renderer_configuration': self.configuration,
            'base_fps': self.baseFps,
            'default_params': self.defaultParams,
            'last_breathing_state': self.last_breathing_state,
            'pending_breathing_state': self.pending_breathing_state,
        }))
        
    def deserialize(self, source: str):
        data = pickle.loads(pathlib.Path(source).read_bytes())
        self.rendererStateCacher.deserialize(data['renderer_state_cacher'])
        self.configuration = data['renderer_configuration']
        self.baseFps = data['base_fps']
        self.defaultParams = data['default_params']
        self.last_breathing_state = data['last_breathing_state']
        self.pending_breathing_state = data['pending_breathing_state']
        
    def on(self, event_name: str, callback: typing.Callable):
        if event_name in self.available_events:
            self.available_events[event_name].append(callback)
        else:
            raise ValueError(f"Event {event_name} is not available.")
        
    def off(self, event_name: str, callback: typing.Callable):
        if event_name in self.available_events:
            self.available_events[event_name].remove(callback)
        else:
            raise ValueError(f"Event {event_name} is not available.")
        
    def trigger_event(self, event_name: str, *args, **kwargs):
        if event_name in self.available_events:
            for callback in self.available_events[event_name]:
                callback(*args, **kwargs)
        else:
            raise ValueError(f"Event {event_name} is not available.")
        
    def compose_animation_group(self, group: list[dict]) -> typing.List[tha4_api.api.PoseUpdateParams]:
        frame_count = 0
        animation_duration = 0 # in seconds
        for composition in group:
            # count total duration of the composition
            current_composition_duration = composition.get("duration", 0) + composition.get("kick_off_offset", 0)
            match composition.get("restore", "reverse"):
                case "reverse":
                    current_composition_duration += composition.get("duration")
                case "rapid":
                    current_composition_duration += composition.get("duration") / 2.0
                case "none":
                    pass
                case _:
                    raise ValueError(f"Invalid restore value: {composition.get('restore')}")
            
            animation_duration = max(animation_duration, current_composition_duration)
        
        # count total number of frames
        total_frames = int(animation_duration * self.baseFps)
        composed: list[tha4_api.api.PoseUpdateParams] = [self.defaultParams.copy() for i in range(total_frames)]
        
        for composition in group:
            start_frame = int(composition.get("kick_off_offset", 0) * self.baseFps)
            end_frame = start_frame + int(composition.get("duration", 0) * self.baseFps)

            match composition.get("action", ""):
                case "eyebrow":
                    animationProvider = tha4_api.animationProvider.LinearAnimation
                    match composition.get("transition", "linear"):
                        case "linear":
                            animationProvider = tha4_api.animationProvider.LinearAnimation
                        case "sine":
                            animationProvider = tha4_api.animationProvider.SineAnimation
                        case "quadratic":
                            animationProvider = tha4_api.animationProvider.QuadraticAnimation
                        case _:
                            raise ValueError(f"Invalid transition value: {composition.get('transition')}")
                            
                    vec_x = animationProvider(
                        composition['x'][0], composition['x'][1], 
                        composition['duration'], self.baseFps)
                    vec_y = animationProvider(
                        composition['y'][0], composition['y'][1], 
                        composition['duration'], self.baseFps)
                    
                    for i in range(start_frame, end_frame):
                        composed[i].set_eyebrow_params(
                            composition.get("desired_state", "normal"), 
                            vec_x[i - start_frame], vec_y[i - start_frame])
                    
                    reverse_x = [i for i in reversed(vec_x)]
                    reverse_y = [i for i in reversed(vec_y)]
                    
                    match composition.get("restore", "reverse"):
                        case "reverse":
                            restore_start_frame = end_frame
                            restore_end_frame = end_frame + (end_frame - start_frame)
                            for i in range(restore_start_frame, restore_end_frame):
                                composed[i].set_eyebrow_params(
                                    composition.get("desired_state", "normal"), 
                                    reverse_x[i - end_frame], reverse_y[i - end_frame])
                        case "rapid":
                            restore_start_frame = end_frame
                case "eye":
                    animationProvider = tha4_api.animationProvider.LinearAnimation
                    match composition.get("transition", "linear"):
                        case "linear":
                            animationProvider = tha4_api.animationProvider.LinearAnimation
                        case "sine":
                            animationProvider = tha4_api.animationProvider.SineAnimation
                        case "quadratic":
                            animationProvider = tha4_api.animationProvider.QuadraticAnimation
                        case _:
                            raise ValueError(f"Invalid transition value: {composition.get('transition')}")
                            
                    vec_x = animationProvider(
                        composition['x'][0], composition['x'][1], 
                        composition['duration'], self.baseFps)
                    vec_y = animationProvider(
                        composition['y'][0], composition['y'][1], 
                        composition['duration'], self.baseFps)
                    
                    for i in range(start_frame, end_frame):
                        composed[i].set_eye_params(
                            composition.get("desired_state", "normal"), 
                            vec_x[i - start_frame], vec_y[i - start_frame])
                    
                    reverse_x = [i for i in reversed(vec_x)]
                    reverse_y = [i for i in reversed(vec_y)]
                    
                    match composition.get("restore", "reverse"):
                        case "reverse":
                            restore_start_frame = end_frame
                            restore_end_frame = end_frame + (end_frame - start_frame)
                            for i in range(restore_start_frame, restore_end_frame):
                                composed[i].set_eye_params(
                                    composition.get("desired_state", "normal"), 
                                    reverse_x[i - end_frame], reverse_y[i - end_frame])
                        case "rapid":
                            restore_start_frame = end_frame
                case "iris_rotation":
                    match composition.get("transition", "linear"):
                        case "linear":
                            delta_y_per_frame = (composition['y'][1] - composition["y"][0]) / (end_frame - start_frame)
                            delta_x_per_frame = (composition['x'][1] - composition["x"][0]) / (end_frame - start_frame)
                            
                            for i in range(start_frame, end_frame):
                                composed[i].set_iris_rotation(
                                    composition['y'][0] + delta_y_per_frame * (i - start_frame), 
                                    composition['x'][0] + delta_x_per_frame * (i - start_frame))
                            
                            match composition.get("restore", "reverse"):
                                case "reverse":
                                    restore_start_frame = end_frame
                                    restore_end_frame = end_frame + (end_frame - start_frame)
                                    for i in range(restore_start_frame, restore_end_frame):
                                        composed[i].set_iris_rotation(
                                            composition['y'][1] - delta_y_per_frame * (i - end_frame), 
                                            composition['x'][1] - delta_x_per_frame * (i - end_frame))
                                case "rapid":
                                    restore_start_frame = end_frame
                                    restore_end_frame = int(end_frame + (end_frame - start_frame) / 2.0)
                                    restore_end_frame = restore_end_frame if restore_end_frame < len(composed) else len(composed) - 1
                                    
                                    for i in range(restore_start_frame, restore_end_frame):
                                        composed[i].set_iris_rotation(
                                            composition['y'][1] - delta_y_per_frame * 2 * (i - end_frame), 
                                            composition['x'][1] - delta_x_per_frame * 2 * (i - end_frame))
                                case "none":
                                    end_frame = end_frame if end_frame < len(composed) else len(composed) - 1
                                    composed[end_frame].set_iris_rotation(
                                        self.defaultParams.iris_rotation_y, 
                                        self.defaultParams.iris_rotation_x)
                                case _:
                                    raise ValueError(f"Invalid restore value: {composition.get('restore')}")
                        case "sine":
                            vec_x = tha4_api.animationProvider.SineAnimation(
                                composition['x'][0], composition['x'][1], 
                                composition['duration'], self.baseFps)
                            vec_y = tha4_api.animationProvider.SineAnimation(
                                composition['y'][0], composition['y'][1], 
                                composition['duration'], self.baseFps)
                            for i in range(start_frame, end_frame):
                                composed[i].set_iris_rotation(
                                    vec_y[i - start_frame], vec_x[i - start_frame])
                            
                            reverse_x = [i for i in reversed(vec_x)]
                            reverse_y = [i for i in reversed(vec_y)]
                            
                            match composition.get("restore", "reverse"):
                                case "reverse":
                                    restore_start_frame = end_frame
                                    restore_end_frame = end_frame + (end_frame - start_frame)
                                    for i in range(restore_start_frame, restore_end_frame):
                                        composed[i].set_iris_rotation(
                                            reverse_y[i - end_frame], reverse_x[i - end_frame])
                                case "rapid":
                                    restore_start_frame = end_frame
                                    restore_end_frame = int(end_frame + (end_frame - start_frame) / 2.0)
                                    restore_end_frame = restore_end_frame if restore_end_frame < len(composed) else len(composed) - 1
                                    
                                    for i in range(restore_start_frame, restore_end_frame):
                                        composed[i].set_iris_rotation(
                                            reverse_y[(i - end_frame) * 2], reverse_x[(i - end_frame) * 2]) 
                        case _:
                            raise ValueError(f"Invalid transition value: {composition.get('transition')}")
                        
                case 'head_rotation':
                    animationProvider = tha4_api.animationProvider.LinearAnimation
                    match composition.get("transition", "linear"):
                        case "linear":
                            animationProvider = tha4_api.animationProvider.LinearAnimation
                        case "sine":
                            animationProvider = tha4_api.animationProvider.SineAnimation
                        case "quadratic":
                            animationProvider = tha4_api.animationProvider.QuadraticAnimation
                        case _:
                            raise ValueError(f"Invalid transition value: {composition.get('transition')}")
                            
                    vec_x = animationProvider(
                        composition['x'][0], composition['x'][1], 
                        composition['duration'], self.baseFps)
                    vec_y = animationProvider(
                        composition['y'][0], composition['y'][1], 
                        composition['duration'], self.baseFps)
                    vec_z = animationProvider(
                        composition['z'][0], composition['z'][1], 
                        composition['duration'], self.baseFps)
                    
                    for i in range(start_frame, end_frame):
                        composed[i].set_head_rotation(vec_x[i - start_frame], 
                            vec_y[i - start_frame], vec_z[i - start_frame])
                    
                    reverse_x = [i for i in reversed(vec_x)]
                    reverse_y = [i for i in reversed(vec_y)]
                    reverse_z = [i for i in reversed(vec_z)]
                    
                    match composition.get("restore", "reverse"):
                        case "reverse":
                            restore_start_frame = end_frame
                            restore_end_frame = end_frame + (end_frame - start_frame)
                            restore_end_frame = restore_end_frame if restore_end_frame < len(composed) else len(composed) - 1
                            
                            for i in range(restore_start_frame, restore_end_frame):
                                composed[i].set_head_rotation(reverse_x[i - end_frame],
                                    reverse_y[i - end_frame], reverse_z[i - end_frame])
                        case "rapid":
                            restore_start_frame = end_frame
                            restore_end_frame = int(end_frame + (end_frame - start_frame) / 2.0)
                            restore_end_frame = restore_end_frame if restore_end_frame < len(composed) else len(composed) - 1
                            
                            for i in range(restore_start_frame, restore_end_frame):
                                composed[i].set_head_rotation(reverse_x[(i - end_frame) * 2],
                                    reverse_y[(i - end_frame) * 2], reverse_z[(i - end_frame) * 2])
                        case "none":
                            end_frame = end_frame if end_frame < len(composed) else len(composed) - 1
                            composed[end_frame].set_head_rotation(
                                self.defaultParams.head_x,
                                self.defaultParams.head_y, 
                                self.defaultParams.neck_z)
                        case _:
                            raise ValueError(f"Invalid restore value: {composition.get('restore')}")
                case "body_rotation":
                    animationProvider = tha4_api.animationProvider.LinearAnimation
                    match composition.get("transition", "linear"):
                        case "linear":
                            animationProvider = tha4_api.animationProvider.LinearAnimation
                        case "sine":
                            animationProvider = tha4_api.animationProvider.SineAnimation
                        case "quadratic":
                            animationProvider = tha4_api.animationProvider.QuadraticAnimation
                        case _:
                            raise ValueError(f"Invalid transition value: {composition.get('transition')}")
                            
                    vec_y = animationProvider(
                        composition['y'][0], composition['y'][1], 
                        composition['duration'], self.baseFps)
                    vec_z = animationProvider(
                        composition['z'][0], composition['z'][1], 
                        composition['duration'], self.baseFps)
                    
                    for i in range(start_frame, end_frame):
                        composed[i].set_body_rotation(
                            vec_y[i - start_frame], vec_z[i - start_frame])
                    
                    reverse_z = [i for i in reversed(vec_z)]
                    reverse_y = [i for i in reversed(vec_y)]
                    
                    match composition.get("restore", "reverse"):
                        case "reverse":
                            restore_start_frame = end_frame
                            restore_end_frame = end_frame + (end_frame - start_frame)
                            for i in range(restore_start_frame, restore_end_frame):
                                composed[i].set_body_rotation(
                                    reverse_y[i - end_frame], reverse_z[i - end_frame])
                        case "rapid":
                            restore_start_frame = end_frame
                            restore_end_frame = int(end_frame + (end_frame - start_frame) / 2.0)
                            restore_end_frame = restore_end_frame if restore_end_frame < len(composed) else len(composed) - 1
                            
                            for i in range(restore_start_frame, restore_end_frame):
                                composed[i].set_body_rotation(
                                    reverse_y[(i - end_frame) * 2], reverse_z[(i - end_frame) * 2])
                        case "none":
                            end_frame = end_frame if end_frame < len(composed) else len(composed) - 1
                            composed[end_frame].set_body_rotation(
                                self.defaultParams.body_y, 
                                self.defaultParams.body_z)
                        case _:
                            raise ValueError(f"Invalid restore value: {composition.get('restore')}")
                case "mouth":
                    every_mouth_state_duration = composition.get("duration", 0) / len(composition.get("shapes", []))
                    for i, shape in enumerate(composition.get("shapes", [])):
                        start_frame_of_shape = start_frame + int(i * every_mouth_state_duration * self.baseFps)
                        end_frame_of_shape = start_frame_of_shape + int(every_mouth_state_duration * self.baseFps)
                        for n in composed[start_frame_of_shape:end_frame_of_shape]: 
                            n.set_active_mouth_shape(shape, True)
                        
                    match composition.get("restore", "reverse"):
                        case "reverse":
                            for i, shape in enumerate(composition.get("shapes", [])):
                                start_frame_of_shape = end_frame + int(i * every_mouth_state_duration * self.baseFps)
                                end_frame_of_shape = start_frame_of_shape + int(every_mouth_state_duration * self.baseFps)
                                for n in composed[start_frame_of_shape:end_frame_of_shape]: 
                                    n.set_active_mouth_shape(shape, True)
                        case "rapid":
                            for i, shape in enumerate(composition.get("shapes", [])):
                                start_frame_of_shape = end_frame + int(i * every_mouth_state_duration * self.baseFps)
                                end_frame_of_shape = start_frame_of_shape + int(every_mouth_state_duration * self.baseFps)
                                for n in composed[start_frame_of_shape:end_frame_of_shape]: 
                                    n.set_active_mouth_shape(shape, True)
                        case "none":
                            end = end_frame + int(i * every_mouth_state_duration * self.baseFps)
                            composed[end if end < len(composed) else len(composed) - 1].set_active_mouth_shape(self.defaultParams.mouth_active_shape, True)
                        case _:
                            raise ValueError(f"Invalid restore value: {composition.get('restore')}")
                
                case "iris_small":
                    animationProvider = tha4_api.animationProvider.LinearAnimation
                    match composition.get("transition", "linear"):
                        case "linear":
                            animationProvider = tha4_api.animationProvider.LinearAnimation
                        case "sine":
                            animationProvider = tha4_api.animationProvider.SineAnimation
                        case "quadratic":
                            animationProvider = tha4_api.animationProvider.QuadraticAnimation
                        case _:
                            raise ValueError(f"Invalid transition value: {composition.get('transition')}")
                            
                    vec_x = animationProvider(
                        composition['x'][0], composition['x'][1], 
                        composition['duration'], self.baseFps)
                    vec_y = animationProvider(
                        composition['y'][0], composition['y'][1], 
                        composition['duration'], self.baseFps)
                    
                    for i in range(start_frame, end_frame):
                        composed[i].set_iris_small(
                            vec_x[i - start_frame], vec_y[i - start_frame])
                    
                    reverse_x = [i for i in reversed(vec_x)]
                    reverse_y = [i for i in reversed(vec_y)]
                    
                    match composition.get("restore", "reverse"):
                        case "reverse":
                            restore_start_frame = end_frame
                            restore_end_frame = end_frame + (end_frame - start_frame)
                            for i in range(restore_start_frame, restore_end_frame):
                                composed[i].set_iris_small(
                                    reverse_x[i - end_frame], reverse_y[i - end_frame])
                        case "rapid":
                            restore_start_frame = end_frame
                            restore_end_frame = int(end_frame + (end_frame - start_frame) / 2.0)
                            restore_end_frame = restore_end_frame if restore_end_frame < len(composed) else len(composed) - 1
                            
                            for i in range(restore_start_frame, restore_end_frame):
                                composed[i].set_iris_small(
                                    reverse_x[(i - end_frame) * 2], reverse_y[(i - end_frame) * 2])
                        case "none":
                            end_frame = end_frame if end_frame < len(composed) else len(composed) - 1
                            composed[end_frame].set_iris_small(
                                self.defaultParams.iris_small_x, 
                                self.defaultParams.iris_small_y)
                case _:
                    raise ValueError(f"Invalid action value: {composition.get('action')}")
        
        # apply breathing
        """
        if self.configuration.breathing:
            breathing_params = self.configuration.breathing
            
            start_frame = len(self.pending_breathing_state)
            frame_len_per_cycle = int(breathing_params['duration'] * self.baseFps)
            end_frame = (int((len(composed) - start_frame) / frame_len_per_cycle) + 1) * frame_len_per_cycle
            
            if len(self.pending_breathing_state) > len(composed):
                for index, val in enumerate(self.pending_breathing_state):
                    if index >= len(composed):
                        break
                    composed[index].set_breathing(val)
                self.pending_breathing_state = self.pending_breathing_state[:len(composed)]
                return composed
            
            for index, val in enumerate(self.pending_breathing_state):
                composed[index].set_breathing(val)
            self.pending_breathing_state = []
            
            # sin(2 pi / duration * (frame - start_frame))
            current_frame_value = lambda x: math.sin(2 * math.pi / frame_len_per_cycle * (x - start_frame))
            for i in range(start_frame, end_frame):
                if i >= len(composed):
                    self.pending_breathing_state.append(current_frame_value(i))
                else:
                    composed[i].set_breathing(current_frame_value(i))
        """
        
        return composed
    
    def compose_state(self, state: str) -> typing.List[tha4_api.api.PoseUpdateParams]:
        if state not in self.configuration.states:
            raise ValueError(f"State {state} is not defined in the configuration.")
        state_params = self.configuration.states[state]
        group = random.choice(state_params)
        
        return self.compose_animation_group(group)
    
    def render_animation(self, composed: typing.List[tha4_api.api.PoseUpdateParams], on_live = False, debugging = False):
        if not composed:
            return
        
        imgs = []
        import time
        current_time = time.time()
        
        for i, frame in enumerate(composed):
            img = self.rendererStateCacher.get_state(frame)
            if debugging:
                import cv2
                imgs.append(img)
            self.trigger_event('frame_update', img)
                        
        if debugging:
            end_time = time.time()
            for i in range(len(imgs)):
                cv2.imshow(f"THA4 API", cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))
                time.sleep(1/(self.baseFps + 10))
                cv2.waitKey(1)
            
            print(f"Composed and rendered {len(imgs)} frames (total {len(imgs) / self.baseFps} seconds) in {end_time - current_time} seconds.")

    def render_animation_no_callback(self, composed: typing.List[tha4_api.api.PoseUpdateParams]) -> typing.List[numpy.ndarray]:
        if not composed:
            return []
        
        self.render_animation(composed, on_live=False, debugging=False)
        imgs = []
        for i, frame in enumerate(composed):
            img = self.rendererStateCacher.get_state(frame)
            imgs.append(img)
        return imgs

    def run_render_loop(self):
        while self.connected:
            try:
                states = self.compose_state(self.pending_states.get(block=False))
                logger.Logger.log('Composing desired state')
                self.render_animation(states)
                time.sleep(len(states) / self.baseFps)
            except queue.Empty:
                logger.Logger.log('Composing idle state')
                states = self.compose_state('idle')
                self.render_animation(states)
                time.sleep(len(states) / self.baseFps)
            
    def start_render_loop(self):
        self.connected = True
        self.render_thread = threading.Thread(target=self.run_render_loop, daemon=True)
        self.render_thread.start()
        
    def stop_render_loop(self):
        self.connected = False
                
                
    def convert_composed_frames_to_mp4(
        self,
        frames: list[numpy.ndarray],
        crf: int = 23, # Constant Rate Factor: 0 (lossless) to 51 (worst quality), 23 is a good default
        preset: str = 'medium' # Encoding speed/compression ratio: 'ultrafast', 'medium', 'slow' etc.
    ) -> bytes:
        """
        Converts a list of numpy.ndarray frames to MP4 bytes using moviepy.

        Args:
            frames (list[numpy.ndarray]): A list of numpy arrays, where each array
                                    is a video frame (H, W, 3) and dtype numpy.uint8.
            baseFps (int): The frames per second for the output video.
            crf (int): Constant Rate Factor for video quality (lower is better quality, larger file size).
            preset (str): Encoding preset for H.264 (e.g., 'medium', 'fast', 'slow').

        Returns:
            bytes: The MP4 video data as bytes.
        """
        baseFps = self.baseFps
        if not frames:
            raise ValueError("The input frames list cannot be empty.")

        # Validate frame dimensions and dtype (optional but good practice)
        first_frame = frames[0]
        if not (first_frame.dtype == numpy.uint8 and first_frame.ndim == 3 and first_frame.shape[2] == 3):
            print(f"Warning: Frames should ideally be (H, W, 3) and numpy.uint8. Got shape {first_frame.shape} and dtype {first_frame.dtype}")
            # moviepy might be more forgiving, but it's good to be aware.

        # Create an ImageSequenceClip from the list of frames
        # moviepy expects RGB (H, W, 3) for its ImageSequenceClip
        clip = ImageSequenceClip(frames, fps=baseFps)

        try:
            temp_file = pathlib.Path(tempfile.gettempdir()) / pathlib.Path(f'{uuid.uuid4().hex}.mp4')
            
            clip.write_videofile(
                temp_file,
                codec='libx264',
                audio_codec='aac', # Even if no audio, it's a common MP4 requirement
                fps=baseFps,
                preset=preset,
                ffmpeg_params=['-crf', str(crf), '-pix_fmt', 'yuv420p'], # ensures compatibility
                logger=None,
            )
            
            res = temp_file.read_bytes()
            temp_file.unlink()
            return res
        except Exception as e:
            print(f"Error during video conversion with moviepy: {e}", file=sys.stderr)
            # Check if ffmpeg is in PATH or if there's a codec issue
            if "ffmpeg" in str(e).lower() or "codec" in str(e).lower():
                print("Please ensure FFmpeg is installed and added to your system's PATH.", file=sys.stderr)
            raise
