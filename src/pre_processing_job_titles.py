from data_extractor.data_getter import DataGetter
import nltk

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

def process_actual_jobs_list():
    unprocessed_job_roles = DataGetter.get_unique_job_roles()
    print(unprocessed_job_roles)

if __name__ == "__main__":
    # unique_jobs = pre_process_job_titles()

    # print(unique_jobs)
    process_actual_jobs_list()