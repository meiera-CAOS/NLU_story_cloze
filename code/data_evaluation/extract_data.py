import numpy as np


def extract_data_from_files(epochs, file_prefix, file_postfix, line_prefix):
    result_list = []
    for i in range(len(epochs)):
        file_name = file_prefix + str(epochs[i]) + file_postfix
        lines = open(file_name, "r")  # opens file for reading
        for line in lines:
            if line.startswith(line_prefix):
                value = line[len(line_prefix):-2]
                result_list.append(float(value))
    return result_list


# training only on validation data, 10fold cross validation
def get_10split_normal_data(epochs):
    file_prefix = "../../data/ours/guet/main_cv_normal_10spl/cv_"
    file_postfix = ".out"
    line_prefix_NC = "average accuracies NC:  ["
    line_prefix_LS = "average accuracies LS:  ["
    line_prefix_FC = "average accuracies FC:  ["
    _10split_NC = extract_data_from_files(epochs=epochs, file_prefix=file_prefix, file_postfix=file_postfix,
                                          line_prefix=line_prefix_NC)
    _10split_LS = extract_data_from_files(epochs=epochs, file_prefix=file_prefix, file_postfix=file_postfix,
                                          line_prefix=line_prefix_LS)
    _10split_FC = extract_data_from_files(epochs=epochs, file_prefix=file_prefix, file_postfix=file_postfix,
                                          line_prefix=line_prefix_FC)
    return _10split_NC, _10split_LS, _10split_FC


# training only on validation data, 10fold cross validation
def get_10split_dropout_data(epochs):
    file_prefix = "../../data/ours/guet/main_cv_drop_10spl/cv_"
    file_postfix = ".out"
    line_prefix_NC = "average accuracies NC:  ["
    line_prefix_LS = "average accuracies LS:  ["
    line_prefix_FC = "average accuracies FC:  ["
    _10split_NC = extract_data_from_files(epochs=epochs, file_prefix=file_prefix, file_postfix=file_postfix,
                                          line_prefix=line_prefix_NC)
    _10split_LS = extract_data_from_files(epochs=epochs, file_prefix=file_prefix, file_postfix=file_postfix,
                                          line_prefix=line_prefix_LS)
    _10split_FC = extract_data_from_files(epochs=epochs, file_prefix=file_prefix, file_postfix=file_postfix,
                                          line_prefix=line_prefix_FC)
    return _10split_NC, _10split_LS, _10split_FC


# training on 90% of validation data and training data with lstm-generated false endings, validation with 10% of validation data
def get_no_crossvalidate_dropout_data(epochs):
    file_prefix = "../../data/ours/guet/no_cv_dropout_0.1/no_cv_"
    line_A = "accuracies "
    line_B = ":  ["

    file_postfix_1 = "_1.out"
    file_postfix_2 = "_2.out"
    file_postfix_3 = "_3.out"

    NC_rep_1, LS_rep_1, FC_rep_1 = get_part_of_no_crossvalidate_data(epochs=epochs, file_prefix=file_prefix,
                                                         file_postfix=file_postfix_1, line_A=line_A, line_B=line_B)
    NC_rep_2, LS_rep_2, FC_rep_2 = get_part_of_no_crossvalidate_data(epochs=epochs, file_prefix=file_prefix,
                                                         file_postfix=file_postfix_2, line_A=line_A, line_B=line_B)
    NC_rep_3, LS_rep_3, FC_rep_3 = get_part_of_no_crossvalidate_data(epochs=epochs, file_prefix=file_prefix,
                                                         file_postfix=file_postfix_3, line_A=line_A, line_B=line_B)

    NC_averaged = np.mean(np.column_stack((NC_rep_1, NC_rep_2, NC_rep_3)), axis=1)
    LS_averaged = np.mean(np.column_stack((LS_rep_1, LS_rep_2, LS_rep_3)), axis=1)
    FC_averaged = np.mean(np.column_stack((FC_rep_1, FC_rep_2, FC_rep_3)), axis=1)
    return NC_averaged, LS_averaged, FC_averaged



# training on 90% of validation data and training data with lstm-generated false endings, validation with 10% of validation data
def get_no_crossvalidate_normal_data(epochs):
    file_prefix = "../../data/ours/guet/no_cv_normal_0.1/no_cv_"
    line_A = "accuracies "
    line_B = ":  ["

    file_postfix_1 = "_1.out"
    file_postfix_2 = "_2.out"
    file_postfix_3 = "_3.out"

    NC_rep_1, LS_rep_1, FC_rep_1 = get_part_of_no_crossvalidate_data(epochs=epochs, file_prefix=file_prefix,
                                                         file_postfix=file_postfix_1, line_A=line_A, line_B=line_B)
    NC_rep_2, LS_rep_2, FC_rep_2 = get_part_of_no_crossvalidate_data(epochs=epochs, file_prefix=file_prefix,
                                                         file_postfix=file_postfix_2, line_A=line_A, line_B=line_B)
    NC_rep_3, LS_rep_3, FC_rep_3 = get_part_of_no_crossvalidate_data(epochs=epochs, file_prefix=file_prefix,
                                                         file_postfix=file_postfix_3, line_A=line_A, line_B=line_B)

    NC_averaged = np.mean(np.column_stack((NC_rep_1, NC_rep_2, NC_rep_3)), axis=1)
    LS_averaged = np.mean(np.column_stack((LS_rep_1, LS_rep_2, LS_rep_3)), axis=1)
    FC_averaged = np.mean(np.column_stack((FC_rep_1, FC_rep_2, FC_rep_3)), axis=1)
    return NC_averaged, LS_averaged, FC_averaged


def get_part_of_no_crossvalidate_data(epochs, file_prefix, file_postfix, line_A, line_B):
    line_prefix_NC = line_A + "NC" + line_B
    line_prefix_LS = line_A + "LS" + line_B
    line_prefix_FC = line_A + "FC" + line_B
    _10split_NC = extract_data_from_files(epochs=epochs, file_prefix=file_prefix, file_postfix=file_postfix,
                                          line_prefix=line_prefix_NC)
    _10split_LS = extract_data_from_files(epochs=epochs, file_prefix=file_prefix, file_postfix=file_postfix,
                                          line_prefix=line_prefix_LS)
    _10split_FC = extract_data_from_files(epochs=epochs, file_prefix=file_prefix, file_postfix=file_postfix,
                                          line_prefix=line_prefix_FC)
    return _10split_NC, _10split_LS, _10split_FC