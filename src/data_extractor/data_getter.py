import os
import csv


class DataGetter:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    TOWN_DATA = CURRENT_DIR + "/../../towns.txt"
    CSV_TRAIN_DATA = CURRENT_DIR + "/../../data/Test_rev1.csv"

    @classmethod
    def get_towns(cls):
        """
        get list of towns above 50000
        :return:list of towns
        """
        with open(cls.TOWN_DATA, 'r') as f:
            cities = []
            for city in f.readlines():
                cities.append(city.lower().replace("\n", ""))
            return cities

    @classmethod
    def get_raw_training_data(cls, header=None):
        """
        Get training data from csv
        :param header: id, Title, FullDesciription ....
        :return:list of all data, or column list of data if title specified
        """
        with open(cls.CSV_TRAIN_DATA, newline='') as f:
            reader = csv.reader(f, delimiter='\n', quotechar='|')
            headers = next(reader)[0].split(",")
            job_test_data = []
            if header:
                try:
                    col_num = headers.index(header)
                except IndexError as e:
                    return e
                for row in reader:
                    job_test_data.append(row[0].split(",")[col_num])
            else:
                for row in reader:
                    job_test_data.append(row[0].split(","))

            return job_test_data


if __name__ == "__main__":
    # EXAMPLE AND MINI TESTS
    # DataGetter.get_raw_training_data("Title")
    print(DataGetter.get_towns())
