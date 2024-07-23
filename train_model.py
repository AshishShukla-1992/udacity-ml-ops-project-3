# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import ml.clean_data
from ml.data import process_data
from ml.model import train_model, inference,compute_model_metrics,process_slices
import logging
import joblib

# log config 
logging.basicConfig(filename='../logs/log',level=logging.INFO,filemode='w')

# Add code to load in the data.
data = clean_data.cleaned_data()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
logging.info("Train Test data split done")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# processing training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
logging.info("Training model")
model = train_model(X_train,y_train)

logging.info("Saving model")
joblib.dump(model,'model/trained_model.joblib')

# Proces the test data with the process_data function.
X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# saving the encoder and labeler
joblib.dump(encoder_test,'model/encoder.joblib')
joblib.dump(lb_test,'model/lb.joblib')

# inferencing
pred = inference(model,X_test)

# calculating metrics
precision, recall, fbeta = compute_model_metrics(y_test,pred)

logging.info(f"Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}")

# performance of model on slices of dat
logging.info("Processing slices")
slice_metrics = process_slices(test, model, cat_features, 'education', encoder, lb)

logging.info(f"slice_metrics: {slice_metrics}")