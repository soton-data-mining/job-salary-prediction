import pandas as pd
import re


def pandas_vector_to_list(pandas_df):
    py_list = [item[0] for item in pandas_df.values.tolist()]
    return py_list


def get_bin(x, n):
    return format(x, 'b').zfill(n)


def get_label_encoded_feature(feature):
    if isinstance(feature, list):  # Input is list
        feature_as_list = feature
    else:  # Input is pandas df
        feature_as_list = pandas_vector_to_list(feature)
    feature_set = list(set(feature_as_list))  # Get unique values
    label_encoded_feature = []
    for item in feature_as_list:
        encoding = feature_set.index(item)
        label_encoded_feature.append(encoding)
    return [label_encoded_feature]


def get_one_hot_encoded_feature(feature, feature_name):
    # One veeeeery hot encoding
    if isinstance(feature, list):  # Input is list
        feature_as_list = feature
    else:  # Input is pandas df
        feature_as_list = pandas_vector_to_list(feature)
    feature_set = set(feature_as_list)  # Get unique values
    one_hot_encoded_features = []
    encoded_labels = []
    for unique_item in feature_set:  # One hot encode values
        encoded_labels.append(feature_name+'_is_'+str(unique_item))
        unique_feature_field = []
        for individual_item in feature_as_list:
            if str(individual_item) == str(unique_item):
                unique_feature_field.append(1)
            else:
                unique_feature_field.append(0)
        one_hot_encoded_features.append(unique_feature_field)
    return (one_hot_encoded_features, encoded_labels)
    # Return list of lists of one hot encoded features


def get_binary_encoded_feature(feature, feature_name):
    if isinstance(feature, list):  # Input is list
        feature_as_list = feature
    else:  # Input is pandas df
        feature_as_list = pandas_vector_to_list(feature)
    unique_feature_list = list(set(feature_as_list))  # Get unique values
    mapping_dict = {item: index for index, item in enumerate(unique_feature_list)}
    binary_feature_length = len(bin(len(unique_feature_list))[2:])  # How many columns needed
    # Amount of columns needed to express this feature
    binary_encoded_feature = []
    column_names = []
    for i in range(0, binary_feature_length):
        column_names.append('bin_'+feature_name+'_'+str(i))
        binary_column = []
        binary_encoded_feature.append(binary_column)
    # Have a list of lists, length, the length is amount of columns needed to express data
    for item in feature_as_list:
        corresponding_integer = mapping_dict[item]
        corresponding_binary = get_bin(corresponding_integer, binary_feature_length)
        for index, digit in enumerate(corresponding_binary):
            binary_encoded_feature[index].append(digit)
    return (binary_encoded_feature, column_names)


def update_location(location_raw, cleaned_location):
    list_cleaned_location = pandas_vector_to_list(cleaned_location)
    list_location_raw = pandas_vector_to_list(location_raw)
    unique_cleaned_location = set(list_cleaned_location)
    update_count = 0
    for index, item in enumerate(list_cleaned_location):  # Loop for town
        if pd.isnull(item):  # Empty item in cleaned town
            match_count = 0
            matched_town = ''
            for town in unique_cleaned_location:  # In each unique town
                if str(town) in str(list_location_raw[index]):
                    match_count += 1
                    matched_town = town
            if match_count == 1:  # Don't update records that have multiple matches
                update_count += 1
                list_cleaned_location[index] = matched_town
                # print('Found:', list_location_raw[index], ' Updated: ', matched_town)
    # print(update_count)
    return list_cleaned_location


def remove_sub_string(sub_string, feature, remove_preceding_white_space=True,
                      replacement_string=''):
    """
    Auxiliary function for removing a substring from a feature
    E.g. In category: Sales Jobs -> Sales

    :param sub_string: the substring you would like to remove
    :param feature: the feature to remove the substring from
    :param remove_preceding_white_space: Defaults to True
    :param replacement_string: string to use for substitution - defaults to ''
    :return: feature list with the substring removed
    """
    # Guard against non-lists
    if isinstance(feature, list):
        feature_as_list = feature
    else:
        feature_as_list = pandas_vector_to_list(feature)

    # optional whitespace (?)
    white_space_matcher = '\s?'
    if not remove_preceding_white_space:
        white_space_matcher = ''

    # compile re outside of list comprehension
    regex = re.compile(white_space_matcher + sub_string, flags=re.I)

    feature_with_removed_sub_string = [regex.sub(replacement_string, row) if isinstance(row, str)
                                       else ''  # return default string for NANs
                                       for row in feature_as_list]

    return feature_with_removed_sub_string
