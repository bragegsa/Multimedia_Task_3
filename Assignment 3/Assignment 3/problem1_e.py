"""Quantizing, baby"""

import numpy as np


SAMPLES_MEAN = 0
SAMPLES_STD = 1
SAMPLES_SIZE = 10
SAMPLES = np.random.normal(loc=SAMPLES_MEAN, scale=SAMPLES_STD, size=SAMPLES_SIZE)

DELTA = int('00001111', 2)
RANGE = 10

def unform_quantizer(samples, num_of_quantization_levels, quantization_range):
    """Quantzing input values uniformely"""

    delta = np.linspace(start=quantization_range[0], stop=quantization_range[1], num=num_of_quantization_levels)
    values_of_nearest_level = delta * np.floor(samples/delta + 1/2)

    bits_per_sample = np.log2(num_of_quantization_levels)

    return values_of_nearest_level, bits_per_sample


values_of_nearest_level_1, bits_per_sample_1 = unform_quantizer(samples=SAMPLES, num_of_quantization_levels=2, quantization_range=(0,1))
# values_of_nearest_level_2, bits_per_sample_2 = unform_quantizer(samples=SAMPLES, num_of_quantization_levels=4, quantization_range=10)
# values_of_nearest_level_3, bits_per_sample_3 = unform_quantizer(samples=SAMPLES, num_of_quantization_levels=8, quantization_range=10)

print(SAMPLES, "are quantized to", values_of_nearest_level_1, "with", bits_per_sample_1, "bits per sample.\n")
# print(SAMPLES, "are quantized to", values_of_nearest_level_2, "with", bits_per_sample_2, "bits per sample.\n")
# print(SAMPLES, "are quantized to", values_of_nearest_level_3, "with", bits_per_sample_3, "bits per sample.\n")

