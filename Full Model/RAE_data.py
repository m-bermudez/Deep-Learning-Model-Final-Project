import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from RAE_utils import create_sequences

def preprocess_data(df_train, df_test, config):
    df_train = df_train[['SUBJECT_ID', 'HADM_ID', 'GLC']]
    df_test = df_test[['SUBJECT_ID', 'HADM_ID', 'GLC', 'TEXT']]

    scaler = MinMaxScaler()
    train_glc_scaled = scaler.fit_transform(df_train[['GLC']])
    test_glc_scaled = scaler.transform(df_test[['GLC']])

    np.save("scaler.npy", scaler)

    X_train = create_sequences(train_glc_scaled, config.sequence_length)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(X_train_tensor, batch_size=config.batch_size, shuffle=True)

    test_sequences = []
    test_meta = []

    for i in range(len(test_glc_scaled) - config.sequence_length):
        seq = test_glc_scaled[i:i + config.sequence_length]
        test_sequences.append(seq)

        meta = {
            "SUBJECT_ID": df_test.iloc[i + config.sequence_length - 1]['SUBJECT_ID'],
            "HADM_ID": df_test.iloc[i + config.sequence_length - 1]['HADM_ID'],
            "GLC": df_test.iloc[i + config.sequence_length - 1]['GLC'],
            "TEXT": df_test.iloc[i + config.sequence_length - 1]['TEXT'],
        }
        test_meta.append(meta)

    X_test_tensor = torch.tensor(np.array(test_sequences), dtype=torch.float32)
    test_loader = torch.utils.data.DataLoader(X_test_tensor, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader, X_test_tensor.shape[2], test_meta
