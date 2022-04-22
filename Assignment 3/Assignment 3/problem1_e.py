"""Quantizing, baby"""

import numpy as np


# Generate samples from normal distribution
SAMPLES_MEAN = 0
SAMPLES_STD = 1
SAMPLES_SIZE = 10
SAMPLES = np.round(np.random.normal(loc=SAMPLES_MEAN, scale=SAMPLES_STD, size=SAMPLES_SIZE), decimals=3)


def uniform_quantizer(samples, bits_per_sample, quantization_range):
    """Quantzing input values uniformely"""

    delta = (quantization_range[1]-quantization_range[0]) / (2**bits_per_sample-1)
    i = np.clip(np.round((samples-quantization_range[0]) / delta), a_min=quantization_range[0], a_max=quantization_range[1])
    values_of_nearest_level = quantization_range[0] + i*delta

    return np.round(values_of_nearest_level, decimals=3), bits_per_sample


def find_distortion(input_values, quantized_values):

    distortion = np.mean((input_values-quantized_values)**2)

    return np.round(distortion, decimals=2)


def print_quantization_results(x):

    for r in range(5,1,-1):
        
        print("From 1 to 3 bits per sample in the range", -r, "to", r, "\n")
        
        for bitrate in range(1,4):
            values_of_nearest_level_1, bits_per_sample_1 = uniform_quantizer(samples=x, bits_per_sample=bitrate, quantization_range=(-r,r))

            print(SAMPLES, "are quantized to", values_of_nearest_level_1, "with", bits_per_sample_1, "bits per sample.")
            print("The distortion is", find_distortion(input_values=x, quantized_values=values_of_nearest_level_1), "\n")

        print("\n")


print_quantization_results(x=SAMPLES)