import os


class DataGetter:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    STOP_WORDS_WITH_CITIES = CURRENT_DIR + "/../../preprocessing_data/stop_word_with_cities.txt"
    TOWN_DATA = CURRENT_DIR + "/../../preprocessing_data/towns.txt"
    CSV_TRAIN_DATA = CURRENT_DIR + "/../../data/Test_rev1.csv"
    UNIQUE_JOB_DATA = CURRENT_DIR + "/../../preprocessing_data/job_roles_unique.txt"

    @staticmethod
    def _get_data_from_file(filename, lowercase=True):
        with open(filename, 'r') as f:
            read_data = []
            for row in f.readlines():
                row_data = row.lower() if lowercase else row
                read_data.append(row_data.replace("\n", ""))
            return read_data

    @classmethod
    def get_towns(cls, lowercase=True):
        """
        get list of towns above 10000
        :return:list of towns
        """
        return DataGetter._get_data_from_file(cls.TOWN_DATA, lowercase)

    @classmethod
    def get_stop_word_inc_cities(cls, lowercase=True):
        """
        get list of towns above 10000 pop + stop words
        :return:list of towns
        """
        return DataGetter._get_data_from_file(cls.STOP_WORDS_WITH_CITIES, lowercase)

    @classmethod
    def get_unique_job_roles(cls, lowercase=True):
        """
        Get list of unique jobs as specifed by british achieve
        """
        return DataGetter._get_data_from_file(cls.UNIQUE_JOB_DATA, lowercase)
