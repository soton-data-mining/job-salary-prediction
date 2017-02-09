import os
import csv
from bisect import bisect
from src.helpers.stop_words import get_stopwords
from collections import Counter

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CSV_TRAIN_DATA = CURRENT_DIR + "/../../data/Test_rev1.csv"
UNIQUE_JOBS_DATA = CURRENT_DIR + "/../../data/unique_jobs.txt"



def get_jobs_from_file()

def count_occurances_of_words(header, file=CSV_TRAIN_DATA):
    with open(file, newline='') as f:
        reader= csv.reader(f, delimiter='\n', quotechar='|')
        jobs = []
        i = 0
        for row in reader:
            i+=1
            jobs.append(str(row).lower())
            if i>10000:
                break

        all_words = []
        job_string = " ".join(jobs)
        word_counter = Counter(job_string)
        for item in word_counter.items(): print("{}\t{}".format(*item))

def remove_stop_words():
    stop_words = get_stopwords()

def read_in_unique_jobs():
    with open(UNIQUE_JOBS_DATA, 'r') as f:
        text = f.readline()
        #Remove white space, and make lower case
        unique_jobs = [job.lower().replace('"', "").strip() for job in text.split(",")]
        #Depluralise
        unique_jobs = [job[:-1] if job[-1]=="s" else job for job in unique_jobs]
        return unique_jobs

if __name__ == "__main__":
    unique_jobs = read_in_unique_jobs()
    count_occurances_of_words("Title")
