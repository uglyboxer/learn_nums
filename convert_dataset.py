""" Convert the dataset into a binary dataset """

def convert_dataset(dataset):
	return [[0 if elem < 5 else 1 for elem in dataset[x]] for x in range(0, len(dataset))]

if __name__ == '__main__':
	print("Test this")