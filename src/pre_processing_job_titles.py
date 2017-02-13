from data_extractor.data_getter import DataGetter
import nltk

def pre_process_job_titles():
    nltk.download('punkt')
    cached_stop_words = DataGetter.get_stop_word_inc_cities()
    job_titles = DataGetter.get_raw_training_data("Title")
    job_title_processed = []
    title = nltk.word_tokenize(job_titles[i])
    removed_stop_title = " ".join([word for word in title if word not in cached_stop_words])
    job_title_processed.append(removed_stop_title)
    return job_title_processed

def process_actual_jobs_list():
    unprocessed_job_roles = DataGetter.get_unique_job_roles()
    set_of_unique_jobs = set()
    set_of_unique_modifiers = set()
    for job in unprocessed_job_roles:
        job_parts = [job_part.strip() for job_part in job.split(",")]
        set_of_unique_jobs.add(job_parts[0])
        for word in job_parts[1:]:
            set_of_unique_modifiers.add(word)
    return set_of_unique_jobs, set_of_unique_modifiers

if __name__ == "__main__":
    # unique_jobs = pre_process_job_titles()
    # print(unique_jobs)
    a,b = process_actual_jobs_list()