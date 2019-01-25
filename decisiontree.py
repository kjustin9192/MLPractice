import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from math import log2

# Training, validation, test dataset.
x_training = []
y_training = []
x_validation = []
y_validation = []
x_test = []
y_test = []

cv = CountVectorizer()

feature0_key = ""  # This variable is to check whether 'compute_information_gain()' method operates well.


def load_data():
    """ We want to
    1. load data
    2. preprocess it using vectorizer (sklearn.feature_extraction.text)
    3. split entire dataset randomly into
      - 70% training
      - 15% validation
      - 15% test examples.
      and store the splitted dataset into global variables <x_training>, <y_training>, <x_validation>, <y_validation>,
      <x_test> and <y_test>.
    """
    real = open("clean_real.txt")  # each line: in list
    fake = open("clean_fake.txt")  # each line: collections.OrderedDict

    # Load data on <dataset> and <label> where <dataset> is list of string headlines and <label> is list of
    # 0 or 1 which corresponds to fake or real headlines, respectively.
    temp_dataset = []
    dataset = []
    label = []

    # Merge input data with target data before shuffling, so that we can shuffle input and target together.
    for line in real:
        temp_dataset.append([line, "1"])
    for line in fake:
        temp_dataset.append([line, "0"])

    # Random shuffling
    random.shuffle(temp_dataset)

    # Divide back to input data and target data.
    for lst in temp_dataset:
        dataset.append(lst[0])
        label.append(lst[1])

    # Preprocess <dataset>
    global cv
    preprocessed = cv.fit_transform(dataset)  # sparse matrix

    # Split preprocessed dataset into 70% training, 15% validation and 15% test set.
    global x_training, y_training, x_validation, y_validation, x_test, y_test
    x_training, y_training, x_validation, y_validation, x_test, y_test \
        = my_split_dataset(preprocessed, label, 0.7, 0.15)


def select_model():
    """
    Train decision tree with at least 5 different max-depth, as well as two different split criteria.
    Print the resulting accuracy of each model.
    """
    # Decision tree classier of two different criterion, each tree having at least 5 different max_depth.
    clf_gini_1 = DecisionTreeClassifier(max_depth=2)
    clf_gini_2 = DecisionTreeClassifier(max_depth=4)
    clf_gini_3 = DecisionTreeClassifier(max_depth=8)
    clf_gini_4 = DecisionTreeClassifier(max_depth=16)
    clf_gini_5 = DecisionTreeClassifier(max_depth=32)
    clf_gini_6 = DecisionTreeClassifier(max_depth=64)
    clf_entropy_1 = DecisionTreeClassifier(criterion="entropy", max_depth=2)
    clf_entropy_2 = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    clf_entropy_3 = DecisionTreeClassifier(criterion="entropy", max_depth=8)
    clf_entropy_4 = DecisionTreeClassifier(criterion="entropy", max_depth=16)
    clf_entropy_5 = DecisionTreeClassifier(criterion="entropy", max_depth=32)
    clf_entropy_6 = DecisionTreeClassifier(criterion="entropy", max_depth=64)

    # Fit training data
    clf_gini_1 = clf_gini_1.fit(x_training, y_training)
    clf_gini_2 = clf_gini_2.fit(x_training, y_training)
    clf_gini_3 = clf_gini_3.fit(x_training, y_training)
    clf_gini_4 = clf_gini_4.fit(x_training, y_training)
    clf_gini_5 = clf_gini_5.fit(x_training, y_training)
    clf_gini_6 = clf_gini_6.fit(x_training, y_training)
    clf_entropy_1 = clf_entropy_1.fit(x_training, y_training)
    clf_entropy_2 = clf_entropy_2.fit(x_training, y_training)
    clf_entropy_3 = clf_entropy_3.fit(x_training, y_training)
    clf_entropy_4 = clf_entropy_4.fit(x_training, y_training)
    clf_entropy_5 = clf_entropy_5.fit(x_training, y_training)
    clf_entropy_6 = clf_entropy_6.fit(x_training, y_training)

    # Prediction
    y_predict_gini_1 = clf_gini_1.predict(x_validation)
    y_predict_gini_2 = clf_gini_2.predict(x_validation)
    y_predict_gini_3 = clf_gini_3.predict(x_validation)
    y_predict_gini_4 = clf_gini_4.predict(x_validation)
    y_predict_gini_5 = clf_gini_5.predict(x_validation)
    y_predict_gini_6 = clf_gini_6.predict(x_validation)
    y_predict_ent_1 = clf_entropy_1.predict(x_validation)
    y_predict_ent_2 = clf_entropy_2.predict(x_validation)
    y_predict_ent_3 = clf_entropy_3.predict(x_validation)
    y_predict_ent_4 = clf_entropy_4.predict(x_validation)
    y_predict_ent_5 = clf_entropy_5.predict(x_validation)
    y_predict_ent_6 = clf_entropy_6.predict(x_validation)

    # Result
    # Validation is checked through helper function 'validation_check()'
    print("Validation check using Gini coefficient. Max depth: 2, 4, 8, 16, 32, 64:")
    print(validation_check(y_predict_gini_1, y_validation))
    print(validation_check(y_predict_gini_2, y_validation))
    print(validation_check(y_predict_gini_3, y_validation))
    print(validation_check(y_predict_gini_4, y_validation))
    print(validation_check(y_predict_gini_5, y_validation))
    print(validation_check(y_predict_gini_6, y_validation))
    print("\nValidation check using Information Gain. Max depth: 2, 4, 8, 16, 32, 64:")
    print(validation_check(y_predict_ent_1, y_validation))
    print(validation_check(y_predict_ent_2, y_validation))
    print(validation_check(y_predict_ent_3, y_validation))
    print(validation_check(y_predict_ent_4, y_validation))
    print(validation_check(y_predict_ent_5, y_validation))
    print(validation_check(y_predict_ent_6, y_validation))

    # Visualization using graphviz
    export_graphviz(clf_gini_6, out_file="gini.dot", max_depth=2, filled=True)
    export_graphviz(clf_entropy_6, out_file="IG.dot", max_depth=2, filled=True)

    # Store the key of root node from <clf_entropy_6> decision tree classifier to 'feature0_key' global variable,
    # in order to pass it as a parameter to 'compute_information_gain()' method and check
    # if the method operates well.
    global feature0_key
    feature0_index = clf_entropy_6.tree_.feature[0]
    d = cv.vocabulary_
    for key in d.keys():
        if d[key] == feature0_index:
            feature0_key = key
            return


def compute_information_gain(feature: str):
    """
    If word <feature> exists in either of headlines, this function prints out the information gain for the
    topmost decision tree split, otherwise prints out "Feature doesn't exist".

    Note that
    - the training data that we want to work on are available as global attribute.
    Here, our input training data is vectorized.
    - load_data() method should be called before we use this method.
    """
    available_features = cv.vocabulary_.keys()
    if feature not in available_features:
        print("Feature doesn't exist")
        return
    index = cv.get_feature_names().index(feature)  # index of <feature> in vectorized training input data.

    n_total = len(y_training)  # Number of total input
    n_label1 = 0  # Number of inputs with label 1
    n_feature_label1 = 0  # Number of inputs that has <feature> and label 1
    n_feature_label0 = 0  # Number of inputs that has <feature> and label 0
    n_nofeature_label1 = 0  # Number of inputs that doesn't have <feature> and label 1
    n_nofeature_label0 = 0  # Number of inputs that doesn't have <feature> and label 0

    for i in range(len(y_training)):
        if y_training[i] == '1':
            n_label1 += 1
            if x_training[i].indices.__contains__(index):
                n_feature_label1 += 1
            else:
                n_nofeature_label1 += 1
        else:
            if x_training[i].indices.__contains__(index):
                n_feature_label0 += 1
            else:
                n_nofeature_label0 += 1

    n_feature = n_feature_label0 + n_feature_label1  # Number of inputs that has <feature>
    n_nofeature = n_nofeature_label0 + n_nofeature_label1  # Number of inputs that doesn't have <feature>

    p_feature = n_feature / (n_feature + n_nofeature)
    p_nofeature = 1 - p_feature

    # Let Y = 0 or 1 and let X = feature or no feature
    p_label1 = 1.0 * n_label1 / n_total  # P(Y=1)
    p_label0 = 1 - p_label1  # P(Y=0)

    p_feature_label1 = 1.0 * n_feature_label1 / (n_feature_label0 + n_feature_label1)  # P(Y=1|X=feature)
    p_feature_label0 = 1.0 - p_feature_label1  # P(Y=0|X=feature)

    p_nofeature_label1 = 1.0 * n_nofeature_label1 / (n_nofeature_label0 + n_nofeature_label1)  # P(Y=1|X=no feature)
    p_nofeature_label0 = 1.0 - p_nofeature_label1  # P(Y=0|X=no feature)

    ent_root = entropy(p_label0) + entropy(p_label1)  # entropy H(Y)
    ent_feature = entropy(p_feature_label0) + entropy(p_feature_label1)  # H(Y|X=feature)
    ent_no_feature = entropy(p_nofeature_label0) + entropy(p_nofeature_label1)  # H(Y|X=no feature)

    information_gain = ent_root - (p_feature * ent_feature + p_nofeature * ent_no_feature)

    print("\nCompute information gain with keyword '" + feature + "':")
    print(information_gain)


def my_split_dataset(X, y, training_percentage, validation_percentage):
    """
    A helper function.

    Returns 6 lists e.g. (X1, y1, X2, y2, X3, y3) where each list corresponds to input and label
    of training, validation and test data.
    * note that percentage of test data equals to 1 - <training> - <validation>.

    X: Sparse matrix
    y: list of labels
    training: percentage of training data
    validation: percentage of validation data
    test: percentage of test data

    Return: tuple of list
    """
    len_training = round(len(y) * training_percentage)
    len_validation = round(len(y) * validation_percentage)

    return (
        X[: len_training],
        y[: len_training],
        X[len_training: len_training + len_validation],
        y[len_training: len_training + len_validation],
        X[len_training + len_validation:],
        y[len_training + len_validation:]
    )


def validation_check(y_prediction, y):
    """
    A helper function.

    Returns the rate(float type) which represents correct prediction rate
    y_prediction: list of prediction values
    y: true label values
    """
    count = 0
    for i in range(len(y_prediction)):
        if y_prediction[i] == y[i]:
            count += 1
    return 1.0 * count / len(y_prediction)


def entropy(x):
    """
    A helper function that returns entropy value.
    """
    if x == 0:
        return 0
    return -1 * x * log2(x)


if __name__ == '__main__':
    load_data()
    select_model()
    compute_information_gain(feature0_key)
    compute_information_gain("trump")
    compute_information_gain("hillary")
