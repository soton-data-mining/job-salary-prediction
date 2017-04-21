import csv

import numpy as np
from sklearn.model_selection import train_test_split
from cleaning_functions import (get_one_hot_encoded_feature,
                                get_binary_encoded_feature,
                                update_location,
                                pandas_vector_to_list,
                                remove_sub_string)
from company_feature_extraction import clean_company_name
from job_description_feature_extraction import get_one_hot_encoded_words
from job_titles_feature_extraction import get_stemmed_sentences


class FeatureObject:
    def __init__(self):
        # you can add/cache other data (like raw description) by adding it to the object

        # TODO Preprocessing should be done here so we don't need all this =None declarations
        self.onehot_desc_words = None
        self.onehot_contract_type = None
        self.onehot_contract_time = None
        self.binary_company = None
        self.binary_source = None
        self.binary_town = None
        self.binary_region = None
        self.binary_job_titles = None
        self.binary_job_modifiers = None
        self.binary_categories = None
        self.salary = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def get_encoded_data(self):
        concatenated_features = np.concatenate((
            self.onehot_desc_words,
            self.onehot_contract_type,
            self.onehot_contract_time,
            self.binary_company,
            self.binary_source,
            self.binary_town,
            self.binary_region,
            self.binary_job_titles,
            self.binary_job_modifiers,
            self.binary_categories,
            self.salary
        ))
        return np.transpose(concatenated_features).astype(int)

    def split_training_test(self, train_size=0.75, test_size=None, force_new_split=False):
        # reuse split from other models so we are using exactly the same training
        # and test data if reusing feature objects
        if force_new_split or not hasattr(self, 'x_train') or self.x_train is None:
            train_data, test_data = train_test_split(self.get_encoded_data(),
                                                     train_size=train_size,
                                                     test_size=test_size,
                                                     random_state=1)
            # Because 1 is good
            # Random state is there so that train and test is always the same for everyone
            self.x_train = train_data[:, 0:train_data.shape[1] - 1]
            self.y_train = train_data[:, train_data.shape[1] - 1]

            self.x_test = test_data[:, 0:test_data.shape[1] - 1]
            self.y_test = test_data[:, test_data.shape[1] - 1]
        else:
            print("reusing existing training split")

    def export_csv(self, filename):
        """
        not needed anymore, since we use pickle, but if we want to look at the data, we can export
        """
        # you can add arbitrary variables here, key will be the column name
        self._export_csv(filename, onehot_desc_words=self.onehot_desc_words,
                         onehot_contract_type=self.onehot_contract_type,
                         onehot_contract_time=self.onehot_contract_time,
                         binary_company=self.binary_company,
                         binary_source=self.binary_source,
                         binary_town=self.binary_town,
                         binary_region=self.binary_region,
                         binary_job_titles=self.binary_job_titles,
                         binary_job_modifiers=self.binary_job_modifiers,
                         binary_categories=self.binary_categories,
                         salary=self.salary)

    def _export_csv(self, filename, **kwargs):
        headers = []
        csv_write = []
        # TODO: maybe want to do something to preserve the order here
        for column_name, data in kwargs.items():
            if isinstance(data, list):
                for i, sublist in enumerate(data):
                    headers.append(column_name + str(i))
                    csv_write.append(sublist)
            else:
                raise NotImplementedError("please implement an exporter for this type")
        writer = csv.writer(open(filename, 'w'))
        writer.writerow(headers)
        writer.writerows(zip(*csv_write))

    def preprocess_data(self, data, normalized_location_data):
        # Get features to clean and reshape
        description_feature = data[['FullDescription']]
        contract_type_feature = data[['ContractType']]
        contract_time_feature = data[['ContractTime']]
        category_feature = data[['Category']]
        company_feature = data[['Company']]
        source_name_feature = data[['SourceName']]
        location_raw_feature = data[['LocationRaw']]
        title_feature = data[['Title']]
        salary_feature = data[['SalaryNormalized']]

        # Read cleaned locations from separate file, cleaned with Google location
        cleaned_town_feature = normalized_location_data[['town']]
        cleaned_region_feature = normalized_location_data[['region']]
        print('Pre-processing begins')
        # Description: consisting of 5 features existence of words below in the description:
        # 'excellent', 'graduate', 'immediate', 'junior', 'urgent'
        onehot_encoded_desc_words = get_one_hot_encoded_words(description_feature)
        # Contract type: One hot encoded, 3 features: part time, full time, *is empty*
        onehot_encoded_contract_type = get_one_hot_encoded_feature(contract_type_feature)
        # Contract time: One hot encoded, 3 features: permanent, contract, *is empty*
        onehot_encoded_contract_time = get_one_hot_encoded_feature(contract_time_feature)
        # Company: Binary encoded
        cleaned_company = clean_company_name(company_feature)
        binary_encoded_company = get_binary_encoded_feature(cleaned_company)
        # Source name: Binary encoded
        binary_encoded_source = get_binary_encoded_feature(source_name_feature)
        # Town: Binary encoded
        print('Pre-processing halfway done')
        updated_town_feature = update_location(location_raw_feature,
                                               cleaned_town_feature)
        binary_encoded_town = get_binary_encoded_feature(updated_town_feature)

        # Region: Binary encoded
        updated_region_feature = update_location(location_raw_feature,
                                                 cleaned_region_feature)
        binary_encoded_region = get_binary_encoded_feature(updated_region_feature)
        # Job titles and modifiers: Binary encoded
        processed_job_titles, processed_job_modifiers = get_stemmed_sentences(title_feature)
        binary_encoded_job_titles = get_binary_encoded_feature(processed_job_titles)
        binary_encoded_job_modifiers = get_binary_encoded_feature(processed_job_modifiers)
        print(len(set(processed_job_titles)))
        print(len(set(processed_job_modifiers)))
        print(len(set(title_feature)))
        # Job category: Binary encoded
        processed_category_feature = remove_sub_string('Jobs', category_feature)
        binary_encoded_categories = get_binary_encoded_feature(processed_category_feature)

        print('Pre-processing ends \n')

        self.onehot_desc_words = onehot_encoded_desc_words
        self.onehot_contract_type = onehot_encoded_contract_type
        self.onehot_contract_time = onehot_encoded_contract_time
        self.binary_company = binary_encoded_company
        self.binary_source = binary_encoded_source
        self.binary_town = binary_encoded_town
        self.binary_region = binary_encoded_region
        self.binary_job_titles = binary_encoded_job_titles
        self.binary_job_modifiers = binary_encoded_job_modifiers
        self.binary_categories = binary_encoded_categories
        self.salary = [pandas_vector_to_list(salary_feature)]
