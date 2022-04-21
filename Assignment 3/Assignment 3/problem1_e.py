''' Quantizing '''

import numpy as np

quantization_levels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

def unform_quantizer(input_value, quantization_levels):
    
    value_of_nearest_level = quantization_levels[np.argmin(np.abs(input_value-quantization_levels))]

    return value_of_nearest_level

input_value = 4.3

print(input_value, "is quantized to", unform_quantizer(input_value=input_value, quantization_levels=quantization_levels))