# -*- coding: utf-8 -*-
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from CATS.inputs import get_feature_names
from CATS.models.dnn_model import *

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    data = pd.read_csv("./example/criteo_example.txt")

    sparse_features = ["C" + str(i) for i in range(1, 27)]
    dense_features = ["I" + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna("-1")
    data[dense_features] = data[dense_features].fillna(0)
    target = ["label"]

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
        for feat in sparse_features
    ] + [DenseFeat(feat, 1) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = "cpu"
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print("cuda ready...")
        device = "cuda:0"

    dnn_model = DNNModel(linear_feature_columns, dnn_feature_columns)
    dnn_model.compile("adam", "binary_cross_entropy", metrics=["log_loss", "auc"])

    history = dnn_model.fit(
        train_model_input,
        train[target].values,
        batch_size=32,
        epochs=10,
        verbose=2,
        validation_split=0.2,
    )
