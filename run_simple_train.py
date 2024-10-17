import pandas as pd
import torch.cuda
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from CATS.inputs import SparseFeat, DenseFeat, get_feature_names

if __name__ == "__main__":
    # chunks = pd.read_csv('../ml_Datasets/criteo_kaggle/train.txt',
    #                    nrows=1000000,
    #                    chunksize=100000,
    #                    low_memory=False)


    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]

    col_names_train = ['Label'] + dense_features + sparse_features
    col_names_test = col_names_train[1:]

    data = pd.read_csv('../ml_Datasets/criteo_kaggle/train.txt',
                     sep='\t',
                     names=col_names_train,
                     nrows=100,
                     low_memory=False)

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1. Label Encoding for sparse features, and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()                        # numeric class 로 바꾸기 위해 작성
        data[feat] = lbe.fit_transform(data[feat])  # numeric class 로 변환
    mms = MinMaxScaler(feature_range=(0, 1))        # 0~1사이의 값으로 변환.
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2. count #unique features for each sparse field, and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max()+1, embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, 1,)
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2024)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model, train, predict and eval

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('using CUDA [cuda:0]')
        device = 'cuda:0'



