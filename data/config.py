def get_dataset_path(dataset_name: str):
    """
    Args:
        dataset_name: 'Solar', 'Wind'
    Return:
        (path_train, path_test, entire_mean, entire_std)
    """
    if dataset_name == 'Solar':
        # mean and std of the training dataset
        train_mean = '392.8659294288083,125.10559238383577'
        train_std = '351.102247720423,101.6698946847449'
    elif dataset_name == 'Wind':
        train_mean = '-0.6822469972863247,-1.258876065025364'
        train_std = '6.004449308199222,4.82805202508728'
    else:
        raise ValueError(f"Please provide mean and std for normalizing {dataset_name} dataset.")

    path_train = f"data/{dataset_name}/train"
    path_test = f"data/{dataset_name}/test"
    return (path_train, path_test, train_mean, train_std)
