''' Quantizing '''

import numpy as np

RANDOM_INPUT = np.round(np.random.rand(10) * 10, 1)
DELTA = int('00001111', 2)
DELTA = 1
RANGE = 10

def unform_quantizer(input_values, delta):
    """Quanitzing input values uniformely"""

    # value_of_nearest_level = quantization_levels[np.argmin(np.abs(input_value-quantization_levels))]

    value_of_nearest_level = delta * np.floor(input_values/delta + 1/2)

    return value_of_nearest_level


print(RANDOM_INPUT, "is quantized to", unform_quantizer(input_values=RANDOM_INPUT, delta=DELTA))