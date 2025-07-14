import PIL.Image
import torch
import numpy
from tha4.shion.base.image_util import extract_pytorch_image_from_PIL_image, pytorch_rgba_to_numpy_image, \
    pytorch_rgb_to_numpy_image
from tha4.image_util import grid_change_to_numpy_image, resize_PIL_image
from tha4.poser.poser import Poser
from typing import Optional

import torch
from typing import List, Dict
from tha4.poser.poser import PoseParameterGroup, PoseParameterCategory


class PoseUpdateParams:
    """
    Represents and manages the state of all adjustable pose parameters,
    mirroring the controls available in the GUI. This version consolidates
    eyebrow and eye parameters to reflect that only one morph type can be
    active per category at a time.

    - Continuous Parameters (sliders): Stored as floats, typically in a [0.0, 1.0] range.
    - Discrete Parameters (checkboxes): Stored as booleans. `True` means active.
    - Choice Parameters (dropdowns): Stored as strings representing the active choice.
    """

    def __init__(self):
        """
        Initializes all pose parameters with default values.
        These defaults are based on the initial state of the wxPython GUI components.
        """
        # --- Morph Categories with Dropdowns ---
        self.eyebrow_choice: str = "troubled"
        self.eyebrow_values: dict[str, float] = {'left': 0.0, 'right': 0.0}

        self.eye_choice: str = "eye_wink"
        self.eye_values: dict[str, float] = {'left': 0.0, 'right': 0.0}

        # --- Discrete Morph Category (Mouth) ---
        self.mouth_active_shape: str = "mouth_aaa"
        self.mouth_is_active: bool = True

        # --- Simple Morph Category (Iris) ---
        self.iris_small_values: dict[str, float] = {'left': 0.0, 'right': 0.0}

        # --- Rotation and Other Simple Sliders (normalized [0.0, 1.0]) ---
        # Default value is 0.5, representing the center of the slider (0.0 in a [-1, 1] range).
        
        # Iris Rotation
        self.iris_rotation_x: float = 0.5
        self.iris_rotation_y: float = 0.5

        # Head Rotation
        self.head_x: float = 0.5
        self.head_y: float = 0.5
        self.neck_z: float = 0.5

        # Body Rotation
        self.body_y: float = 0.5
        self.body_z: float = 0.5

        # Breathing
        self.breathing: float = 0.0


    def copy(self) -> 'PoseUpdateParams':
        """
        Returns a deep copy of the pose parameters.
        """
        new_params = PoseUpdateParams()
        new_params.eyebrow_choice = self.eyebrow_choice
        new_params.eyebrow_values = self.eyebrow_values.copy()
        new_params.eye_choice = self.eye_choice
        new_params.eye_values = self.eye_values.copy()
        new_params.mouth_active_shape = self.mouth_active_shape
        new_params.mouth_is_active = self.mouth_is_active
        new_params.iris_small_values = self.iris_small_values.copy()
        new_params.iris_rotation_x = self.iris_rotation_x
        new_params.iris_rotation_y = self.iris_rotation_y
        new_params.head_x = self.head_x
        new_params.head_y = self.head_y
        new_params.neck_z = self.neck_z
        new_params.body_y = self.body_y
        new_params.body_z = self.body_z
        new_params.breathing = self.breathing
        return new_params


    # --- Getters and Setters for Consolidated Parameters ---

    def set_eyebrow_params(self, choice: str, left_value: float, right_value: float):
        """Sets the active eyebrow morph and its slider values."""
        self.eyebrow_choice = choice
        self.eyebrow_values = {'left': left_value, 'right': right_value}

    def get_eyebrow_params(self) -> tuple[str, dict[str, float]]:
        """Gets the active eyebrow morph and its slider values."""
        return self.eyebrow_choice, self.eyebrow_values

    def set_eye_params(self, choice: str, left_value: float, right_value: float):
        """Sets the active eye morph and its slider values."""
        self.eye_choice = choice
        self.eye_values = {'left': left_value, 'right': right_value}

    def get_eye_params(self) -> tuple[str, dict[str, float]]:
        """Gets the active eye morph and its slider values."""
        return self.eye_choice, self.eye_values

    # --- Getters and Setters for Other Parameters ---

    def set_active_mouth_shape(self, shape_name: str, is_active: bool):
        """Sets the active discrete mouth shape."""
        self.mouth_active_shape = shape_name
        self.mouth_is_active = is_active

    def get_active_mouth_shape(self) -> tuple[str, bool]:
        """Gets the active discrete mouth shape and its state."""
        return self.mouth_active_shape, self.mouth_is_active

    def set_iris_small(self, left_value: float, right_value: float):
        """Sets the iris small morph slider values."""
        self.iris_small_values = {'left': left_value, 'right': right_value}

    def get_iris_small(self) -> dict[str, float]:
        """Gets the iris small morph slider values."""
        return self.iris_small_values

    def set_iris_rotation(self, y: float, x: float):
        self.iris_rotation_y = y
        self.iris_rotation_x = x

    def get_iris_rotation(self) -> tuple[float, float]:
        return self.iris_rotation_y, self.iris_rotation_x
    
    def set_head_rotation(self, x: float, y: float, z: float):
        """Sets head rotation values (head_x, head_y, neck_z)."""
        self.head_x = x
        self.head_y = y
        self.neck_z = z

    def get_head_rotation(self) -> tuple[float, float, float]:
        """Gets head rotation values."""
        return self.head_x, self.head_y, self.neck_z

    def set_body_rotation(self, y: float, z: float):
        """Sets body rotation values (body_y, body_z)."""
        self.body_y = y
        self.body_z = z

    def get_body_rotation(self) -> tuple[float, float]:
        """Gets body rotation values."""
        return self.body_y, self.body_z


    def set_breathing(self, value: float):
        self.breathing = value

    def get_breathing(self) -> float:
        return self.breathing
    
    def __repr__(self):
        return f"PoseUpdateParams(eyebrow_choice='{self.eyebrow_choice}', eyebrow_values={self.eyebrow_values}, " \
               f"eye_choice='{self.eye_choice}', eye_values={self.eye_values}, " \
               f"mouth_active_shape='{self.mouth_active_shape}', mouth_is_active={self.mouth_is_active}, " \
               f"iris_small_values={self.iris_small_values}, iris_rotation_x={self.iris_rotation_x}, " \
               f"iris_rotation_y={self.iris_rotation_y}, head_x={self.head_x}, head_y={self.head_y}, " \
               f"neck_z={self.neck_z}, body_y={self.body_y}, body_z={self.body_z}, breathing={self.breathing})"
               
    def __hash__(self):
        return hash(self.__repr__())
    
    def __eq__(self, value):
        if isinstance(value, PoseUpdateParams):
            for attr in self.__dict__:
                if getattr(self, attr) != getattr(value, attr):
                    return False
                return True
        return False
            

class PoseMapper:
    """
    Translates the state from a PoseUpdateParams object into a model-compatible
    pose vector. It acts as the bridge between the high-level parameter representation
    and the low-level numerical vector required by the poser model.
    """

    def __init__(self, pose_parameter_groups: List[PoseParameterGroup], num_parameters: int):
        """
        Initializes the mapper with the model's parameter definitions.

        Args:
            pose_parameter_groups: The list of parameter groups from the loaded poser model.
            num_parameters: The total number of parameters for the final pose vector.
        """
        self.num_parameters: int = num_parameters
        self.group_map: Dict[str, PoseParameterGroup] = {
            group.get_group_name(): group for group in pose_parameter_groups
        }


    @staticmethod
    def _format_name(name: str) -> str:
        """Converts a display name like "Eyebrow Troubled" to a key like "eyebrow_troubled"."""
        return name.lower().replace(" ", "_")

    def create_pose_vector(self, params: PoseUpdateParams) -> List[float]:
        """
        Generates the final pose vector from a PoseUpdateParams object.

        This is the core method that performs the translation.

        Args:
            params: The PoseUpdateParams object holding the current GUI state.

        Returns:
            A list of floats representing the final pose vector for the model.
        """
        # Start with a default pose vector of all zeros.
        pose = [0.0] * self.num_parameters

        # 1. Handle Consolidated Morph: Eyebrow
        eyebrow_choice, eyebrow_values = params.get_eyebrow_params()
        eyebrow_group = self.group_map.get(self._format_name(eyebrow_choice))
        if eyebrow_group:
            param_index = eyebrow_group.get_parameter_index()
            param_range = eyebrow_group.get_range()
            # Apply left slider value
            alpha_left = eyebrow_values['left']
            pose[param_index] = param_range[0] + \
                (param_range[1] - param_range[0]) * alpha_left
            # Apply right slider value if the morph supports it (arity == 2)
            if eyebrow_group.get_arity() == 2:
                alpha_right = eyebrow_values['right']
                pose[param_index + 1] = param_range[0] + \
                    (param_range[1] - param_range[0]) * alpha_right

        # 2. Handle Consolidated Morph: Eye
        eye_choice, eye_values = params.get_eye_params()
        eye_group = self.group_map.get(self._format_name(eye_choice))
        if eye_group:
            param_index = eye_group.get_parameter_index()
            param_range = eye_group.get_range()
            alpha_left = eye_values['left']
            pose[param_index] = param_range[0] + \
                (param_range[1] - param_range[0]) * alpha_left
            if eye_group.get_arity() == 2:
                alpha_right = eye_values['right']
                pose[param_index + 1] = param_range[0] + \
                    (param_range[1] - param_range[0]) * alpha_right

        # 3. Handle Discrete Morph: Mouth
        mouth_shape, is_active = params.get_active_mouth_shape()
        if is_active:
            mouth_group = self.group_map.get(self._format_name(mouth_shape))
            if mouth_group and mouth_group.is_discrete():
                param_index = mouth_group.get_parameter_index()
                # Set all parameters for this discrete group to 1.0
                for i in range(mouth_group.get_arity()):
                    pose[param_index + i] = 1.0

        # 4. Handle Simple Morphs: Iris Small
        iris_small_values = params.get_iris_small()
        iris_small_group = self.group_map.get("iris_small")
        if iris_small_group:
            param_index = iris_small_group.get_parameter_index()
            param_range = iris_small_group.get_range()
            pose[param_index] = param_range[0] + \
                (param_range[1] - param_range[0]) * iris_small_values['left']
            if iris_small_group.get_arity() == 2:
                pose[param_index + 1] = param_range[0] + \
                    (param_range[1] - param_range[0]) * \
                    iris_small_values['right']

        # 5. Handle all other Simple Slider-based Parameters
        # This mapping explicitly links the PoseUpdateParams attributes to the group names.
        simple_param_map = {
            "iris_rotation_x": params.iris_rotation_x,
            "iris_rotation_y": params.iris_rotation_y,
            "head_x": params.head_x,
            "head_y": params.head_y,
            "neck_z": params.neck_z,
            "body_y": params.body_y,
            "body_z": params.body_z,
            "breathing": params.breathing
        }

        for name, alpha in simple_param_map.items():
            # 'name' is now a direct key, e.g., "head_x", no formatting needed.
            group = self.group_map.get(name)
            if group:
                param_index = group.get_parameter_index()
                param_range = group.get_range()
                # Calculate the final pose value from the normalized alpha [0.0, 1.0]
                # This correctly translates, for example, alpha=1.0 to a value of 1.0
                # in a [-1.0, 1.0] range.
                pose[param_index] = param_range[0] + (param_range[1] - param_range[0]) * alpha
            else:
                # This warning can help during debugging if a mismatch occurs.
                print(f"Warning: Pose parameter group '{name}' not found in model definition.")

        return pose



def convert_output_image_from_torch_to_numpy(output_image: torch.Tensor) -> numpy.ndarray:
    """Converts a torch tensor output to a numpy array for display."""
    if output_image.shape[2] == 2:
        h, w, c = output_image.shape
        numpy_image = torch.transpose(
            output_image.reshape(h * w, c), 0, 1).reshape(c, h, w)
    elif output_image.shape[0] == 4:
        numpy_image = pytorch_rgba_to_numpy_image(output_image)
    elif output_image.shape[0] == 3:
        numpy_image = pytorch_rgb_to_numpy_image(output_image)
    elif output_image.shape[0] == 1:
        c, h, w = output_image.shape
        alpha_image = torch.cat([output_image.repeat(
            3, 1, 1) * 2.0 - 1.0, torch.ones(1, h, w)], dim=0)
        numpy_image = pytorch_rgba_to_numpy_image(alpha_image)
    elif output_image.shape[0] == 2:
        numpy_image = grid_change_to_numpy_image(output_image, num_channels=4)
    else:
        raise RuntimeError(
            f"Unsupported # image channels: {output_image.shape[0]}")
    return numpy.uint8(numpy.rint(numpy_image * 255.0))


class ImageInferenceManager:
    """
    Handles model loading, base image management, and inference execution.
    """

    def __init__(self, poser_model: Poser, device: torch.device):
        """
        Initializes the inference manager.

        Args:
            poser_model: The pre-loaded poser model.
            device: The torch device (e.g., 'cuda:0' or 'cpu').
        """
        self.poser: Poser = poser_model
        self.device: torch.device = device
        self.dtype: torch.dtype = self.poser.get_dtype()
        self.torch_source_image: Optional[torch.Tensor] = None

    def get_image_size(self) -> int:
        """Returns the image size required by the model."""
        return self.poser.get_image_size()

    def set_base_image(self, pil_image: PIL.Image.Image):
        """
        Sets and preprocesses the base image for inference.

        Args:
            pil_image: A PIL Image object with an RGBA channel.
        """
        if pil_image.mode != 'RGBA':
            raise ValueError("Image must have an alpha (RGBA) channel.")

        resized_image = resize_PIL_image(
            pil_image,
            (self.get_image_size(), self.get_image_size())
        )
        self.torch_source_image = extract_pytorch_image_from_PIL_image(resized_image) \
            .to(self.device) \
            .to(self.dtype)

    def clear_base_image(self):
        """Clears the current base image."""
        self.torch_source_image = None

    def inference(self, pose_params: PoseUpdateParams, output_index: int) -> Optional[numpy.ndarray]:
        """
        Runs the model inference with the given parameters.

        Args:
            pose_params: An object containing the desired pose parameters.
            output_index: The index of the output layer to use.

        Returns:
            A numpy array representing the output image, or None if no base image is set.
        """
        if self.torch_source_image is None:
            return None

        # pose_vector = pose_params.get_pose_vector()
        pose_vector = PoseMapper(self.poser.get_pose_parameter_groups(
        ), self.poser.get_num_parameters()).create_pose_vector(pose_params)
        pose_tensor = torch.tensor(
            pose_vector, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            output_image_tensor = self.poser.pose(
                self.torch_source_image, pose_tensor, output_index)[0]
            output_image_tensor = output_image_tensor.detach().cpu()

        numpy_image = convert_output_image_from_torch_to_numpy(
            output_image_tensor)
        return numpy_image

    @staticmethod
    def load_model(device: torch.device) -> Poser:
        """A static method to load the poser model."""
        import tha4.poser.modes.mode_07
        return tha4.poser.modes.mode_07.create_poser(device)
