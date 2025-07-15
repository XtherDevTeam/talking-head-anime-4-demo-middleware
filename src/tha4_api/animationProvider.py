import math
import typing


def LinearAnimation(begin: float, end: float, duration: float, fps: int) -> typing.List[float]:
    """
    Generate a linear animation from begin to end over duration at fps frames per second.
    """
    frames = int(duration * fps)

    if frames <= 0:
        return []
    if frames == 1:
        return [begin]

    result = []
    for i in range(frames):

        t = i / (frames - 1)

        value = begin + (end - begin) * t
        result.append(value)

    return result


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


def QuadraticAnimation(begin: float, end: float, duration: float, fps: int) -> typing.List[float]:
    """
    Generate a quadratic animation from begin to end over duration at fps frames per second.
    The interpolation follows a quadratic ease-in-out curve, ensuring the animation
    starts and ends smoothly.
    """
    frames = int(duration * fps)
    if frames <= 0:
        return []
    if frames == 1:
        return [begin]

    result = []
    for i in range(frames):

        t = i / (frames - 1)

        factor = 2 * t ** 2 if t < 0.5 else -2 * t ** 2 + 4 * t - 1

        value = begin + (end - begin) * factor
        result.append(value)

    return result

