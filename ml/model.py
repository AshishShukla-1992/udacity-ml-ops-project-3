from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data

# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    return classifier


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    prediction = model.predict(X)
    return prediction


def process_slices(df, model, cat_features, feature, encoder=None, lb=None):
    """
    Function to process the slices of data

    Args:
        df (pd.DataFrame): Cleaned data 
        model (RandomForestClassifier): Trained ML model
        cat_features (List[str]): Categorical Features
        feature (str): Feature for which slices should be calculated
        encoder (sklearn.preprocessing._encoders.OneHotEncoder) :
        Trained sklearn OneHotEncoder, only used if training=False.
        lb (sklearn.preprocessing._label.LabelBinarizer) :
        Trained sklearn LabelBinarizer, only used if training=False.
    Returns:
        slices(List[str]): Metrices of slices
    """
    slices = {}

    for feat_value in df[feature].unique():
        slice_df = df[df[feature] == feat_value]

        # Proces the slice data with the process_data function.
        X_slice_test, y_slice_test, encoder_slice_test, lb_slice_test = process_data(
            slice_df, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

        y_slice_preds = inference(model, X_slice_test)

        dict_metrices = dict(zip(("precision", "recall", "fbeta"),
                             compute_model_metrics(y_slice_test, y_slice_preds)))

        slices[feat_value] = dict_metrices

    with open('slice_metrix.txt', 'w') as f:
        for k, v in slices.items():
            f.write(f"{k}:{v}")
            f.write("\n")

    return slices
