import pandas as pd
from matplotlib import pyplot as plt

input_file = 'data/DDC/unseen/test_formatted.txt'




with open(input_file, 'r') as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]
prefixes = [line.split('\t')[0] for line in lines]


# create histogram of prefix length


prefix_lengths = [len(prefix) for prefix in prefixes]
plt.hist(prefix_lengths, bins=1)
plt.xlabel('Prefix Length')
plt.ylabel('Frequency')
plt.title('Prefix Length Histogram')
plt.savefig('prefix_length_histogram.png')