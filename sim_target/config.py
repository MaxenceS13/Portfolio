import numpy as np

# Dimensions du monde
WORLD_LENGTH = 1000
WORLD_WIDTH = 1000
MAX_DISTANCE = np.sqrt(WORLD_LENGTH**2 + WORLD_WIDTH**2)

# Paramètres de simulation
DT = 0.1
MAX_TIME = 50.0

# Paramètres des entités
MISSILE_SPEED = 50.0
MISSILE_RADIUS = 8.0
TARGET_RADIUS = MISSILE_RADIUS
TARGET_VX = 30.0
TARGET_VY = -10.0
MAX_TURN_ANGLE = np.pi / 12  # ~15° par frame