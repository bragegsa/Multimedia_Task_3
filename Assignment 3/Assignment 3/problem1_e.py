"""Quantizing, baby"""

import numpy as np
import matplotlib.pyplot as plt


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


def find_rate_distortion_function(sample_std):
    """Calculate rate-distortion"""

    distortion = np.linspace(start=0.001, stop=2*sample_std, num=1000)
    rate_distortion_function = 1/2 * np.log2(sample_std**2 / distortion)

    return distortion, np.maximum(rate_distortion_function, 0)


def print_quantization_results(x):
    """Print and plot quantization results"""

    source, rate_distortion_function_source = find_rate_distortion_function(SAMPLES_STD)

    distortions = np.array([])
    rates = np.array([])

    plt.figure()
    plt.plot(source, rate_distortion_function_source)
    plt.grid()

    for r in range(5,1,-1):
        
        print("From 1 to 3 bits per sample in the range", -r, "to", r, "\n")
        print(x, "are quantized to \n")
        
        for bitrate in range(1,4):
            values_of_nearest_level, bits_per_sample = uniform_quantizer(samples=x, bits_per_sample=bitrate, quantization_range=(-r,r))

            print(values_of_nearest_level, "with", bits_per_sample, "bits per sample.")
            print("The distortion is", find_distortion(input_values=x, quantized_values=values_of_nearest_level), "\n")

            distortions = np.append(distortions, find_distortion(input_values=x, quantized_values=values_of_nearest_level))
            rates = np.append(rates, bits_per_sample)
            
        print("\n")
    
    # for i in range(len(distortions)):
    #     plt.plot(distortions[i], rates[i])
    #     plt.show()


print_quantization_results(x=SAMPLES)
