''' Quantizing '''

import numpy as np

QUANTIZATION_LEVELS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

def unform_quantizer(input_value, quantization_levels):
    """Quanitzing input values uniformely"""
    
    value_of_nearest_level = quantization_levels[np.argmin(np.abs(input_value-quantization_levels))]

    return value_of_nearest_level

INPUT_VALUE = 4.3

print(INPUT_VALUE, "is quantized to", unform_quantizer(input_value=INPUT_VALUE, quantization_levels=QUANTIZATION_LEVELS))