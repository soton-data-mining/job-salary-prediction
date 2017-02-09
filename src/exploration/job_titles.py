import os
import csv
from bisect import bisect
from collections import Counter
from src.data_extractor.data_getter import DataGetter
import RAKE

STOP_WORDS_WIHT_CITIES_PATH = "../../stop_word_with_cities.txt"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CSV_TRAIN_DATA = CURRENT_DIR + "/../../data/Test_rev1.csv"
UNIQUE_JOBS_DATA = CURRENT_DIR + "/../../data/unique_jobs.txt"



def pre_process_job_titles():
        job_titles = DataGetter.get_raw_training_data("Title")
        job_title_processed = []
        for i in range(10000):
            stop_word_remover = RAKE.Rake(STOP_WORDS_WIHT_CITIES_PATH)
            pre_processed_job_title = job_titles[i]
            job_title = stop_word_remover.run(str(pre_processed_job_title))
            job_title_processed.append(job_title)
            i +=1
        return job_title_processed


if __name__ == "__main__":
    unique_jobs = pre_process_job_titles()
