from cleaning_functions import (get_one_hot_encoded_feature,
                                get_binary_encoded_feature,
                                update_location,
                                remove_sub_string)
from job_description_feature_extraction import (get_one_hot_encoded_words,
                                                get_rake_keywords,
                                                get_top_idf_features)
from job_titles_feature_extraction import get_stemmed_sentences
from models.BaseModel import BaseModel


class OldMainMethod(BaseModel):
    def predict_salary(self):
        # Update empty town-region features with already existing features
        updated_town_feature = update_location(self.train_location_raw_feature,
                                               self.cleaned_town_feature)
        updated_region_feature = update_location(self.train_location_raw_feature,
                                                 self.cleaned_region_feature)

        # Train one hot encoded features - desciption, contract-type, contract-time
        # All lists containt list of lists of features in one hot encoded format ready to append to
        # cleaned set

        # Description: consisting of 5 features existence of words below in the description:
        # 'excellent', 'graduate', 'immediate', 'junior', 'urgent'
        onehot_encoded_desc_words = get_one_hot_encoded_words(self.train_description_feature)
        # Contract type: One hot encoded, 3 features: part time, full time, *is empty*
        onehot_encoded_contract_type = get_one_hot_encoded_feature(self.train_contract_type_feature)
        # Contract time: One hot encoded, 3 features: permanent, contract, *is empty*
        onehot_encoded_contract_time = get_one_hot_encoded_feature(self.train_contract_time_feature)

        # Train binary encoded features - company, source, town, region
        # Company name: Binary encoded
        train_binary_encoded_company = get_binary_encoded_feature(self.train_company_feature)
        # Source name: Binary encoded
        train_binary_encoded_source = get_binary_encoded_feature(self.train_source_name_feature)
        # Town: Binary encoded
        cleaned_binary_encoded_town = get_binary_encoded_feature(updated_town_feature)
        # Region: Binary encoded
        cleaned_binary_encoded_region = get_binary_encoded_feature(updated_region_feature)

        # get stemmed/sorted job title and job title modifiers

        processed_job_titles, processed_job_modifiers = get_stemmed_sentences(
            self.train_data[['Title']]
        )

        train_binary_encoded_job_titles = get_binary_encoded_feature(processed_job_titles)
        train_binary_encoded_job_modifiers = get_binary_encoded_feature(processed_job_modifiers)

        # Remove superfluous 'Jobs' from job category
        processed_category_feature = remove_sub_string('Jobs', self.train_category_feature)
        # Job category: Binary encoded
        train_binary_encoded_categories = get_binary_encoded_feature(processed_category_feature)

        keywords = get_rake_keywords(self.train_description_feature)
        # one_hot_encoded_features are a list of 5 lists consisting of 1s and 0s
        # If a word appears in the description, this kind of representation will
        # make it feasible for machine learning
        # We will need to append those lists on final data frame when everything is cleaned

        # get top k terms with the highest idf score (this is atm not very useful
        # but it will be used for the tf.idf similarity stuff
        top_keywords = get_top_idf_features(self.train_description_feature, 5)

        # no-op functions to avoid pep8 "unused variable" error
        isinstance(onehot_encoded_desc_words, int)
        isinstance(onehot_encoded_contract_type, int)
        isinstance(onehot_encoded_contract_time, int)
        isinstance(train_binary_encoded_company, int)
        isinstance(train_binary_encoded_source, int)
        isinstance(cleaned_binary_encoded_town, int)
        isinstance(cleaned_binary_encoded_region, int)
        isinstance(train_binary_encoded_job_titles, int)
        isinstance(train_binary_encoded_job_modifiers, int)
        isinstance(keywords, int)
        isinstance(top_keywords, int)
        isinstance(train_binary_encoded_categories, int)

        return [0] * self.test_data_size
