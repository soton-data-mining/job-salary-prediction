from collections import Counter

from cleaning_functions import remove_sub_string, pandas_vector_to_list
from job_description_feature_extraction import get_top_idf_features


def clean_company_name(company_names):
    # 1. remove high scoring idf terms ('stop words' for this specific case)
    feature_as_list = remove_common_terms(company_names)

    # 2. remove/replace special characters
    feature_as_list = remove_special_chars(feature_as_list)

    # 3. strip and lowercase only after we have removed all special chars
    feature_as_list = [x.lower().strip() for x in feature_as_list]

    # debug_output(company_names, feature_as_list)

    return feature_as_list


def debug_output(company_names, feature_as_list):
    counter_clean = Counter(feature_as_list)
    counter_dirty = Counter([x.strip() if isinstance(x, str)
                             else ''  # return default string for NANs
                             for x in pandas_vector_to_list(company_names)])
    print("unique count dirty = {} of which {} are ''".format(
        len(counter_dirty), counter_dirty['']))
    print("unique count clean = {} of which {} are ''".format(
        len(counter_clean), counter_clean['']))
    # save unique company names to txt file
    open('companies.txt', 'w').write(
        '\n'.join(sorted([x + "||" + str(y) for x, y in counter_clean.items() if y > 0])))


def remove_special_chars(company_names):
    """
    replaces special characters from company names

    :param company_names: list of strings
    :return: list of company names with special chars removed/replaced
    """
    regex_remove_special_chars = '([\.&,/\'])'
    regex_replace_special_chars = '[-–]'
    regex_replace_multiple_spaces = '[\s]{2,}'
    feature_as_list = remove_sub_string(regex_remove_special_chars, company_names, False)
    feature_as_list = remove_sub_string(regex_replace_special_chars, feature_as_list, False, " ")
    feature_as_list = remove_sub_string(regex_replace_multiple_spaces, feature_as_list, False, " ")
    return feature_as_list


def remove_common_terms(company_names):
    """
    removes an automatic compiled list of common terms like 'ltd', 'group' and so on
     to group and reduce the number of unique company names

    :param company_names: list of strings
    :return: list of company names with 'stop words' removed
    """
    # this is a mostly automatic compiled list of terms
    # using company_feature_extraction._find_top_idf_words
    # manually removed obvious company names which we do not want to include
    common_terms = ['ltd', 'recruitment', 'limited', 'group', 'services', 'the', 'solutions',
                    'school', 'uk', 'and', 'care', 'of', 'associates', 'consulting', 'resourcing',
                    'personnel', 'search', 'primary', 'council', 'college', 'people', 'it',
                    'selection', 'cleaning', 'london', 'university', 'management', 'international',
                    'home', 'trust', 'plc', 'education', 'resources', 'centre', 'st', 'healthcare',
                    'technical', 'consultancy', 'support', 'hotel', 'street', 'academy',
                    'consultants', 'house', 'company', 'for', 'housing', 'employment',
                    'resource', 'engineering', 'recruit', 'bureau', 'partnership', 'co', 'security',
                    'training', 'health', 'technology', 'global', 'brook', 'business', 'executive',
                    'legal', 'appointments', 'media', 'first', 'service', 'marketing',
                    'association', 'community', 'agency', 'park', 'network', 'financial', 'hr',
                    'sales', 'direct', 'foundation', 'retail', 'professional', 'partners', 'jobs',
                    'nursing', 'society', 'construction', 'staff', 'llp', '.co.uk']
    # watch out for word boundaries \b
    regex_remove_common_terms = r"\b(" + '|'.join(common_terms) + r")\b"
    return remove_sub_string(regex_remove_common_terms, company_names)


def _find_top_idf_words(company_names):
    """
    determines highest scoring idf terms ('stop words' for this specific case)
    used to be remove terms from our company names to reduce the unique count of company names

    :param company_names: list of strings
    :return: list of highest scoring idf terms
    """
    feature_as_list = remove_special_chars(company_names)
    feature_as_list = [x.lower().strip() for x in feature_as_list]
    feature_as_list = set(feature_as_list)
    features = get_top_idf_features(feature_as_list, 100, 1)
    print(features)
    return features
