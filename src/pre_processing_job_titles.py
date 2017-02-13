import os
from src.data_extractor.data_getter import DataGetter
import nltk

STOP_WORDS_WIHT_CITIES_PATH = "../../stop_word_with_cities.txt"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CSV_TRAIN_DATA = CURRENT_DIR + "/../../data/Test_rev1.csv"
UNIQUE_JOBS_DATA = CURRENT_DIR + "/../../data/unique_jobs.txt"


def pre_process_job_titles():
    nltk.download('punkt')
    cached_stop_words = DataGetter.get_stop_word_inc_cities()
    job_titles = DataGetter.get_raw_training_data("Title")
    job_title_processed = []
    for i in range(10000):
        title = nltk.word_tokenize(job_titles[i])
        removed_stop_title = " ".join([word for word in title if word not in cached_stop_words])
        job_title_processed.append(removed_stop_title)
    return job_title_processed


if __name__ == "__main__":
    unique_jobs = pre_process_job_titles()
    print(unique_jobs)