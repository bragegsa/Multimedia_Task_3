"""Testing enumerate"""

CLASSES = ("Fish", "Dog", "Cat")

for i in enumerate(CLASSES):
    plt.plot(recall, precision, label=f'{CLASSES[i]} with an AUPRC of {auprc}')
