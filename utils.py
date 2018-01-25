import math
import numpy as np


def unrotate_positions(relative_positions, rotations):
    new_positions = relative_positions

    # YAW
    yaws = rotations[:, 1]
    yaws = -yaws / 32768. * np.pi

    new_positions[:, 0], new_positions[:, 1] = new_positions[:, 0] * np.cos(yaws) - new_positions[:, 1] * np.sin(yaws), new_positions[:, 0] * np.sin(yaws) + new_positions[:, 1] * np.cos(yaws)

    # PITCH

    pitchs = rotations[:, 0]
    pitchs = pitchs / 32768. * np.pi

    new_positions[:, 2], new_positions[:, 0] = new_positions[:, 2] * np.cos(pitchs) - new_positions[:, 0] * np.sin(pitchs), new_positions[:, 2] * np.sin(pitchs) + new_positions[:, 0] * np.cos(pitchs)

    # ROLL

    rolls = rotations[:, 2]
    rolls = rolls / 32768. * np.pi

    new_positions[:, 1], new_positions[:, 2] = new_positions[:, 1] * np.cos(rolls) - new_positions[:, 2] * np.sin(rolls), new_positions[:, 1] * np.sin(rolls) + new_positions[:, 2] * np.cos(rolls)

    return new_positions


def _unrotate_positions(relative_positions, rotations):
    """OLD"""
    new_positions = relative_positions

    # YAW
    yaws = rotations[:, 1]
    yaws = yaws / 32768. - 1
    yaws[yaws < -1] += 2

    A = new_positions[:, 0]
    B = new_positions[:, 1]
    angles, magnitudes = get_2D_vector(A, B)

    car_angles = -yaws
    # orange goal is positive Y, but negative yaw value initially.

    new_angles = angles - car_angles

    new_positions[:, 0] = magnitudes * np.cos(np.pi * new_angles)
    new_positions[:, 1] = magnitudes * np.sin(np.pi * new_angles)

    # PITCH

    pitchs = rotations[:, 0]
    pitchs = pitchs / 32768. - 1
    pitchs[pitchs < -1] += 2

    A = new_positions[:, 0]
    B = new_positions[:, 2]
    angles, magnitudes = get_2D_vector(A, B)

    car_angles = pitchs

    new_angles = angles - car_angles

    new_positions[:, 0] = magnitudes * np.cos(np.pi * new_angles)
    new_positions[:, 2] = magnitudes * np.sin(np.pi * new_angles)

    # ROLL

    rolls = rotations[:, 2]
    rolls = rolls / 32768. - 1
    rolls[rolls < -1] += 2

    A = new_positions[:, 1]
    B = new_positions[:, 2]
    angles, magnitudes = get_2D_vector(A, B)

    car_angles = -rolls

    new_angles = angles - car_angles

    new_positions[:, 1] = magnitudes * np.cos(np.pi * new_angles)
    new_positions[:, 2] = magnitudes * np.sin(np.pi * new_angles)

    # new_positions[:, 0], new_positions[:, 1] = -new_positions[:, 1], -new_positions[:, 0]

    return new_positions


def get_2D_vector(A, B):
    # returns 1 for back, 0.5 for up, 0 for forward, -0.5 for down, -1 for
    # back.
    angles = np.arctan2(B, A) / np.pi
    magnitudes = np.sqrt(A**2 + B**2)
    return angles, magnitudes


def unrotate_position(car_rotation, relative_position):
    pitch_, yaw_, roll_ = car_rotation
    new_position = relative_position
    new_position = yaw(car_rotation, new_position, yaw_)
    new_position = pitch(car_rotation, new_position, pitch_)
    new_position = roll(car_rotation, new_position, roll_)

    new_position[0], new_position[1] = -new_position[1], -new_position[0]
    return new_position


def get_vector2D(x, y):
    # angle = math.atan2(y, x) / math.pi
    angle = np.arctan2(y, x) / math.pi
    # returns 1 for back, 0.5 for up, 0 for forward, -0.5 for down, -1 for
    # back.
    magnitude = (x**2 + y**2)**0.5
    return angle, magnitude


def pitch(car_rotation, relative_position, pitch):
    # do pitch: (rotation around y) z is vertical axis, x is horizontal axis.
    # pitch is -0.5(up) to -1(forward) to 1(forward) to 0.5(down)
    # NEW: 16384 (up) to 0 (forward) to -16384 (down)

    # turn new pitch into old pitch
    pitch = pitch / 32768 - 1
    if pitch < -1:
        pitch += 2
    # pitch /= 32768
    # get angle

    angle, magnitude = get_vector2D(
        relative_position[0], relative_position[2])

    # convert to -0.5(down) to 0.5(up)
    # if pitch < 0:
    # playerAngle = 1+pitch
    # elif pitch > 0:
    # playerAngle = 1-pitch
    # else: pass
    playerAngle = pitch

    angleNew = angle - playerAngle
    # if angleNew > 1: angleNew -= 2
    # elif angleNew <= -1: angleNew += 2
    # return the range to 1to-1.

    new_position = []
    new_position.append(magnitude * math.cos(math.pi * angleNew))
    new_position.append(relative_position[1])
    new_position.append(magnitude * math.sin(math.pi * angleNew))

    return new_position


def yaw(car_rotation, relative_position, yaw):
    # do yaw: (rotation around z)
    # -0.5(orange goal) to -1(left) to 1(left) to 0.5(blue goal) to 0(right)
    # NEW: 16834 (orange goal) to 0 (left) to -16384  (blue goal) to +-32768
    # (right)

    # convert new yaw to old yaw
    yaw = yaw / 32768 - 1
    if yaw < -1:
        yaw += 2
    # yaw /= 32768

    (angle, magnitude) = get_vector2D(
        relative_position[0], relative_position[1])

    # playerAngle = -yaw
    playerAngle = -yaw
    # orange goal is positive Y, but negative yaw value initially.

    angleNew = angle - playerAngle

    new_position = []
    new_position.append(magnitude * math.cos(math.pi * angleNew))
    new_position.append(magnitude * math.sin(math.pi * angleNew))
    new_position.append(relative_position[2])

    return new_position


def roll(car_rotation, relative_position, roll):
    # do roll (rotation around x)
    # -0.5(rolled right) to -1(flat) to 1(flat) to  0.5 (left)
    # -0.5 (rolled right) to 0 (upsidedown) to 0.5 (left)
    # NEW: 16384 (rolled right) to 0 (flat) to -16384 (rolled left) to -32768
    # upside down

    # convert new roll to old roll
    roll = roll / 32768 - 1
    if roll < -1:
        roll += 2
    # roll /= 32768

    (angle, magnitude) = get_vector2D(
        relative_position[1], relative_position[2])

    # if roll <= 0:
    # playerAngle = -(roll + 1)
    # elif roll > 0:
    # playerAngle = -(roll - 1)
    # playerAngle = -roll + 1
    playerAngle = -roll
    angleNew = angle - playerAngle

    new_position = []
    new_position.append(relative_position[0])
    new_position.append(magnitude * math.cos(math.pi * angleNew))
    new_position.append(magnitude * math.sin(math.pi * angleNew))

    return new_position
