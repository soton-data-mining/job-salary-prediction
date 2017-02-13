def pandas_vector_to_list(pandas_df):
    py_list = [item[0] for item in pandas_df.values.tolist()]
    return py_list


def get_bin(x, n):
    return format(x, 'b').zfill(n)


def get_one_hot_encoded_feature(feature):
    feature_cleaned = pandas_vector_to_list(feature)
    feature_set = set(feature_cleaned)  # Get unique values

    one_hot_encoded_features = []
    for unique_item in feature_set:  # One hot encode contracts
        unique_feature_field = []
        for individual_item in feature_cleaned:
            if str(individual_item) == str(unique_item):
                unique_feature_field.append(1)
            else:
                unique_feature_field.append(0)
        one_hot_encoded_features.append(unique_feature_field)
    return one_hot_encoded_features  # Return list of lists of one hot encoded features


def get_binary_encoded_feature(feature):
    feature_as_list = pandas_vector_to_list(feature)
    unique_feature_list = list(set(feature_as_list))  # Get unique values

    feature_binary_code = []
    for index, item in enumerate(unique_feature_list):
        feature_binary_code.append(index)
    # A list that has integers on how to represent values
    binary_feature_length = len(bin(len(feature_binary_code))[2:])
    # Amount of columns needed to express this feature
    binary_encoded_feature = []
    for i in range(0, binary_feature_length):
        binary_column = []
        binary_encoded_feature.append(binary_column)
    # Have a list of lists, length, the length is amount of columns needed to express data
    for item in feature_as_list:
        corresponding_integer = feature_binary_code[unique_feature_list.index(item)]
        corresponding_binary = get_bin(corresponding_integer, binary_feature_length)
        for index, digit in enumerate(corresponding_binary):
            binary_encoded_feature[index].append(digit)
    return binary_encoded_feature
