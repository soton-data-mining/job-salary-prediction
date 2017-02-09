def get_one_hot_encoded_feature(feature):
    feature_cleaned = [item[0] for item in feature.values.tolist()]
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
