import math
import typing


def SineAnimation(begin: float, end: float, duration: float, fps: int) -> typing.List[float]:
    """
    Generate a sine animation from begin to end over duration at fps frames per second.
    The interpolation follows a quarter-sine curve (ease-out), ensuring the animation
    starts precisely at 'begin' and ends precisely at 'end'.
    """
    frames = int(duration * fps)

    if frames <= 0:
        return []
    if frames == 1:
        return [begin]

    result = []
    for i in range(frames):

        t = i / (frames - 1)

        factor = math.sin(math.pi / 2 * t)

        value = begin + (end - begin) * factor
        result.append(value)

    return result
