from csv import reader


def loadCsv(filename):
    """ Imports and returns a CSV with a specified file name."""
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
