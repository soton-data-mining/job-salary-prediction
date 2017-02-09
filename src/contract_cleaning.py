def get_one_hot_encoded_contract(contract):
    contract_as_list = contract.values.tolist()  # Returns a list of list [[]]
    contract_cleaned = []

    for item in contract_as_list:
        contract_cleaned.append(item[0])
    contract_set = set(contract_cleaned)  # Get unique values

    one_hot_encoded_contracts = []
    for unique_item in contract_set:  # One hot encode contracts
        unique_contract_field = []
        for individual_item in contract_cleaned:
            if str(individual_item) == str(unique_item):
                unique_contract_field.append(1)
            else:
                unique_contract_field.append(0)
        one_hot_encoded_contracts.append(unique_contract_field)

    return one_hot_encoded_contracts
