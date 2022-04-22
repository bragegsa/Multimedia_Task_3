"""Quantizing, baby"""

import numpy as np


# Generate samples from normal distribution
SAMPLES_MEAN = 0
SAMPLES_STD = 1
SAMPLES_SIZE = 10
SAMPLES = np.random.normal(loc=SAMPLES_MEAN, scale=SAMPLES_STD, size=SAMPLES_SIZE)


def uniform_quantizer(samples, bits_per_sample, quantization_range):
    """Quantzing input values uniformely"""

    delta = (quantization_range[1]-quantization_range[0]) / (2**bits_per_sample-1)
    i = np.clip(np.round((samples-quantization_range[0]) / delta), a_min=quantization_range[0], a_max=quantization_range[1])
    values_of_nearest_level = quantization_range[0] + i*delta

    return values_of_nearest_level, bits_per_sample


# From 1 to 3 bits per sample in the range -3 to 3
values_of_nearest_level_1, bits_per_sample_1 = uniform_quantizer(samples=SAMPLES, bits_per_sample=1, quantization_range=(-3,3))
values_of_nearest_level_2, bits_per_sample_2 = uniform_quantizer(samples=SAMPLES, bits_per_sample=2, quantization_range=(-3,3))
values_of_nearest_level_3, bits_per_sample_3 = uniform_quantizer(samples=SAMPLES, bits_per_sample=3, quantization_range=(-3,3))

print(SAMPLES, "are quantized to", values_of_nearest_level_1, "with", bits_per_sample_1, "bits per sample.\n")
print(SAMPLES, "are quantized to", values_of_nearest_level_2, "with", bits_per_sample_2, "bits per sample.\n")
print(SAMPLES, "are quantized to", values_of_nearest_level_3, "with", bits_per_sample_3, "bits per sample.\n")

# From 1 to 3 bits per sample in the range -4 to 4
values_of_nearest_level_1, bits_per_sample_1 = uniform_quantizer(samples=SAMPLES, bits_per_sample=1, quantization_range=(-4,4))
values_of_nearest_level_2, bits_per_sample_2 = uniform_quantizer(samples=SAMPLES, bits_per_sample=2, quantization_range=(-4,4))
values_of_nearest_level_3, bits_per_sample_3 = uniform_quantizer(samples=SAMPLES, bits_per_sample=3, quantization_range=(-4,4))

print(SAMPLES, "are quantized to", values_of_nearest_level_1, "with", bits_per_sample_1, "bits per sample.\n")
print(SAMPLES, "are quantized to", values_of_nearest_level_2, "with", bits_per_sample_2, "bits per sample.\n")
print(SAMPLES, "are quantized to", values_of_nearest_level_3, "with", bits_per_sample_3, "bits per sample.\n")

# From 1 to 3 bits per sample in the range -2 to 2
values_of_nearest_level_1, bits_per_sample_1 = uniform_quantizer(samples=SAMPLES, bits_per_sample=1, quantization_range=(-2,2))
values_of_nearest_level_2, bits_per_sample_2 = uniform_quantizer(samples=SAMPLES, bits_per_sample=2, quantization_range=(-2,2))
values_of_nearest_level_3, bits_per_sample_3 = uniform_quantizer(samples=SAMPLES, bits_per_sample=3, quantization_range=(-2,2))

print(SAMPLES, "are quantized to", values_of_nearest_level_1, "with", bits_per_sample_1, "bits per sample.\n")
print(SAMPLES, "are quantized to", values_of_nearest_level_2, "with", bits_per_sample_2, "bits per sample.\n")
print(SAMPLES, "are quantized to", values_of_nearest_level_3, "with", bits_per_sample_3, "bits per sample.\n")