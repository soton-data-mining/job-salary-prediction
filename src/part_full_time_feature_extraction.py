import pandas as pd


def get_part_full_time_features(job_describtion_data, part_full_time_data):
    """
    :param job_describtion_data: n number of job descriptions
    :param part_full_time_data:  n number of part_full time job data
    :return: n length list containing 1 or 0. 1 for part time, 0 for full time
    """

    feature_vector = []
    for job in part_full_time_data.iterrows():
        idx = job[0]
        if idx != 0:
            if "part" in str(job[1].values[0]).lower():
                feature_vector.append(1)
            else:
                if "part time" in str(
                        job_describtion_data._values[idx][0]).lower():
                    feature_vector.append(1)
                else:
                    feature_vector.append(0)
    return feature_vector


if __name__ == "__main__":
    TRAIN_NORMALIZED_LOCATION_FILE_NAME = '../data/train_normalised_location.csv'
    TRAIN_DATA_CSV_FILE_NAME = '../data/Train_rev1.csv'
    data = pd.read_csv('../data/Train_rev1.csv')
    part_full_vector = get_part_full_time_features(data[["FullDescription"]],
                                                   data[['ContractType']])
