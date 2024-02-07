
def main():
    pass

if __name__ == "__main__":
    main()



    #
    # X = range(len(original_dataset))
    # y = original_dataset.targets
    #
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.7, stratify=y)
    #
    # idx_to_class = {v: k for k, v in original_dataset.class_to_idx.items()}
    #
    # def get_class_distribution(dataset):
    #     count_dict = {k: 0 for k, v in original_dataset.class_to_idx.items()}  # initialise dictionary
    #
    #     for input, label in dataset:
    #         label = idx_to_class[label]
    #         count_dict[label] += 1
    #
    #     return count_dict
    #
    # k = 6
    # train_dataset = Subset(original_dataset, X_train)
    # # train_dataset, test_dataset = split_dataset(original_dataset, 0.8, 0.2)
    # plt.figure(figsize=(20, 10))
    # sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(train_dataset)]).melt(), x="variable", y="value", hue="variable").set_title('Class Distribution of the dataset')
    # plt.show()