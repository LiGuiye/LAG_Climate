def get_dataset_path(dataset_name: str):
    """
    Args:
        dataset_name: 'Solar_07-14', 'Wind_07-14'
    Return:
        (path_train, path_test, entire_mean, entire_std)
    """
    if dataset_name == 'Solar_07-14':
        path_train = 'data/Solar/train'
        path_test = 'data/Solar/test'
        # mean and std of the training dataset
        entire_mean = '392.8659294288083,125.10559238383577'
        entire_std = '351.102247720423,101.6698946847449'
    elif dataset_name == 'Wind_07-14':
        path_train = 'data/Wind/train'
        path_test = 'data/Wind/test'
        # mean and std of the training dataset
        entire_mean = '-0.6741845839785552,-1.073033474161022'
        entire_std = '5.720375778518578,4.772050058088903'
    return (path_train, path_test, entire_mean, entire_std)
