import pandas as pd
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from mongo import read_mongo, save_mongo
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle
import time
import pymongo

mongo_uri = "mongodb+srv://store:store@cluster0.zc9ptyy.mongodb.net/test"
m_db = "store"
# Press the green button in the gutter to run the script.


coll_list = ["holidays", "oil", "stores", "train", "transactions", "test"]


def train_all_models():
    df_list = []
    for coll in coll_list:
        df = train_model(m_db, coll)
        df_list.append(df)
    return df_list


def train_model(db, coll):
    train_df = read_mongo(mongo_uri, db, coll)
    train_df.head()
    # print(train_df.to_string())
    return train_df


def pre_processing():
    df_list = train_all_models()
    y = df_list[3].groupby(['date', 'family', 'store_nbr'])['sales'].mean().unstack(['family', 'store_nbr'])
    print(y)
    y = y.asfreq('D')
    y.fillna(method='bfill', inplace=True)
    create_features(y, df_list[5], df_list[0])


holidays = ['Local', 'Regional', 'National']


def select_holiday(x):
    return holidays[max([holidays.index(holiday) for holiday in x])]


def create_features(y, test_df, holidays_events_df):
    # Fourier features
    fourier = CalendarFourier(freq='A', order=26)
    # const and trend features
    dp = DeterministicProcess(
        index=y.index,
        constant=True,
        order=1,
        seasonal=True,
        additional_terms=[fourier],
        drop=True,
    )
    X = dp.in_sample()

    holidays_events_trimmed_df = holidays_events_df[holidays_events_df.transferred == False]
    holidays_events_trimmed_df = holidays_events_trimmed_df.set_index('date')
    holidays_events_trimmed_df = holidays_events_trimmed_df['locale']
    holidays_events_trimmed_df = holidays_events_trimmed_df.groupby('date').apply(select_holiday)
    holidays_events_trimmed_df.unique()
    X = X.merge(holidays_events_trimmed_df, how='left', on='date').fillna('NA')
    X['year'] = X.index.year
    X['day_of_week'] = X.index.dayofweek
    X_test = dp.out_of_sample(steps=16)
    X_test = X_test.merge(holidays_events_trimmed_df, how='left', left_index=True, right_on='date').fillna('NA')
    print("printing x test")
    print(X_test)
    X_test.set_index('date', inplace=True)
    X_test['year'] = X_test.index.year
    X_test['day_of_week'] = X_test.index.dayofweek
    X_test.head()
    category_col = ['locale', 'year', 'day_of_week']

    common_df = pd.concat([X[category_col], X_test[category_col]])
    for col in category_col:
        common_df[col] = common_df[col].astype('category')
    dummies_df = pd.get_dummies(common_df, drop_first=True)
    X = pd.concat([X, dummies_df.iloc[:len(X)]], axis=1)
    X_test = pd.concat([X_test, dummies_df.iloc[-len(X_test):]], axis=1)
    X.drop(columns=category_col, inplace=True)
    X_test.drop(columns=category_col, inplace=True)
    X.fillna(method='bfill', inplace=True)
    X_test.fillna(method='bfill', inplace=True)
    model = XGBRegressor(random_state=42)
    params = {'n_estimators': [5, 10, 15, 20, 50, 80, 100],
              'learning_rate': [1e-3, 1e-2, 1e-1, 5e-1, 1],
              'min_child_weight': [1, 3, 5, 7, 10, 20],
              'colsample_bytree': [0.2, 0.3, 0.5, 0.8, 1.0],
              'max_depth': [5, 10, 15, 25, 50]
              }
    random_search = RandomizedSearchCV(model,
                                       param_distributions=params,
                                       n_iter=1,
                                       scoring='neg_mean_squared_error',
                                       cv=2,
                                       verbose=1)
    random_search.fit(X, y)
    random_search.best_estimator_
    test_df.head()
    y_pred = random_search.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred.reshape((-1, y.shape[1])), index=test_df.groupby('date').mean().index,
                             columns=y.columns)
    y_pred_df = y_pred_df.stack(['family', 'store_nbr']).to_frame('sales').reset_index()
    y_pred_df.head()
    print("prediction")
    print(y_pred_df)
    save_mongo(m_db, y_pred_df)
    # test_df = test_df.reset_index().merge(y_pred_df, how='left', on=['date', 'family', 'store_nbr'])
    # test_df.head()
    # print("---------------------------------final prediction------------------------------------------------")
    # print(test_df)

    # save random_search_to DB


def save_prediction(db, pred_pd):
    save_mongo(mongo_uri=mongo_uri, db=db, pred_pd=pred_pd, collection="sales_prediction")


def save_model_to_db(model, client, db, dbconnection, model_name):
    # pickling the model
    pickled_model = pickle.dumps(model)
    # saving model to mongoDB
    # creating connection
    myclient = pymongo.MongoClient(client)
    # creating database in mongodb
    mydb = myclient[db]
    # creating collection
    mycon = mydb[dbconnection]
    info = mycon.insert_one({model_name: pickled_model, 'name': model_name, 'created_time': time.time()})
    print(info.inserted_id, ' saved with this id successfully!')
    details = {
        'inserted_id': info.inserted_id,
        'model_name': model_name,
        'created_time': time.time()
    }

    return details


def load_saved_model_from_db(model_name, client, db, dbconnection):
    json_data = {}
    # saving model to mongoDB
    # creating connection
    myclient = pymongo.MongoClient(client)
    # creating database in mongodb
    mydb = myclient[db]
    # creating collection
    mycon = mydb[dbconnection]
    data = mycon.find({'name': model_name})
    for i in data:
        json_data = i
    # fetching model from db
    pickled_model = json_data[model_name]
    return pickle.loads(pickled_model)


if __name__ == '__main__':
    pre_processing()
