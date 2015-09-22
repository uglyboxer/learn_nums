

def convert_dataset(dataset):
	""" Convert the dataset into a binary dataset """
	return [[elem for elem in dataset[x]] for x in range(0, len(dataset))] #  0 if elem < 5 else 1

if __name__ == '__main__':
	print("Test this")