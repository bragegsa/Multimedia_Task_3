"""Problem 1d, 1e, 1f, and 1g"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate samples from normal distribution
SAMPLES_MEAN = 0
SAMPLES_STD = 1
SAMPLES_SIZE = 100
SAMPLES = np.random.normal(loc=SAMPLES_MEAN, scale=SAMPLES_STD, size=SAMPLES_SIZE)


def uniform_quantizer(samples, bits_per_sample, quantization_range):
    """Quantizing input values uniformely"""

    delta = (quantization_range[1]-quantization_range[0]) / (2**bits_per_sample-1)
    i = np.clip(np.round((samples-quantization_range[0]) / delta), a_min=np.round(2*quantization_range[0]/delta), a_max=np.round(2*quantization_range[1]/delta))
    values_of_nearest_level = quantization_range[0] + delta*i

    return values_of_nearest_level, bits_per_sample


def find_distortion(input_values, quantized_values):
    """Calculate distortion of quantizer output"""

    distortion = np.mean((input_values-quantized_values)**2)

    return distortion


def find_rate_distortion_function(sample_std, stop_dist):
    """Calculate rate-distortion"""

    distortion = np.linspace(start=0.001, stop=stop_dist, num=1000)
    rate_distortion_function = 1/2 * np.log2(sample_std**2 / distortion)

    return distortion, np.maximum(rate_distortion_function, 0)


def print_quantization_results(x):
    """Print and plot quantization results"""

    distortions = np.array([])
    rates = np.array([])

    # for r in range(6,2,-1):
        
    r = 1

    print("From 1 to 3 bits per sample in the range", -r, "to", r, "\n")
    print(x, "are quantized to \n")
    
    for bitrate in range(1,4):
        values_of_nearest_level, bits_per_sample = uniform_quantizer(samples=x, bits_per_sample=bitrate, quantization_range=(-r,r))
        
        plt.figure()
        plt.title("Quantization with bitrate " + str(bitrate) + ", range +/-" + str(r) + " and " + str(2**bitrate) + " levels")
        plt.plot(SAMPLES)
        plt.plot(values_of_nearest_level)
        plt.grid()
        plt.show()

        # print(values_of_nearest_level, "with", bits_per_sample, "bits per sample.")
        # print("The distortion is", find_distortion(input_values=x, quantized_values=values_of_nearest_level), "\n")

        distortions = np.append(distortions, find_distortion(input_values=x, quantized_values=values_of_nearest_level))
        rates = np.append(rates, bits_per_sample)
        
    print("\n")
    
    source, rate_distortion_function_source = find_rate_distortion_function(SAMPLES_STD, np.max(distortions))

    plt.figure()
    plt.title("Rate-distortion function for the quantizer")
    plt.plot(source, rate_distortion_function_source)
    plt.plot(distortions, rates, ".")
    plt.legend(["Source R(D)", "Quantizer R(D)"])
    plt.grid()
    plt.show()

print_quantization_results(SAMPLES)

# # KOK
# X = np.random.normal(SAMPLES_MEAN, SAMPLES_STD, 100)
# X_2D=np.reshape(X, (X.size, 1))

# def vector_quantization(bit):
#     """this is not kok"""
    
#     n_clusters=2**bit

#     x_2D=np.reshape(X, (X.size, 1))
#     kmeans = KMeans(n_clusters=n_clusters).fit(x_2D)
#     values = kmeans.cluster_centers_.squeeze()
#     labels_pred = kmeans.predict(X_2D)
    
#     return values[labels_pred]

# # ----

# def compare_vector_to_default():
#     """Comparing vector quantizer results to the uniform quantizer"""

#     for bit in range(1,4):
        
#         R_vec = vector_quantization(bit)
#         D_vec = find_distortion(X, R_vec)

#         quantized_results, R_def = uniform_quantizer(samples=SAMPLES, bits_per_sample=bit, quantization_range=(-3,3))
#         D_def = find_distortion(X, quantized_values=quantized_results)

#         plt.plot(D_vec, R_vec)
#         plt.plot(D_def, R_def)

#     plt.show()

# compare_vector_to_default()


# print_quantization_results(x=SAMPLES)
