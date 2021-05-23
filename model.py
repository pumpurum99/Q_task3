if __name__ == "__main__":
    path_train_data = input("Train data file path:")
    path_predict_data = input("Predict data file path:")

    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    train_data = pd.read_csv(path_train_data)
    test_data = pd.read_csv(path_predict_data)

    dataset = train_data.copy()
    target_column = 'target'
    y = dataset[target_column]
    X = dataset.drop([target_column], axis=1)
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    rf_model = RandomForestRegressor(n_estimators=10, random_state=0)
    print("It takes some time...")
    rf_model.fit(train_X, train_y)
    preds_rf = rf_model.predict(val_X)

    final_result = pd.concat([
        test_data,
        pd.DataFrame(rf_model.predict(test_data), columns=['target'])
    ], axis=1)
    final_result.to_csv('file_with_model_predictions.csv')

    print("Finished!!!")


