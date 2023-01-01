import pandas as pd
import pymongo


def _connect_mongo(mongo_uri, db):
    """ A util for making a connection to mongo """
    # set a 5-second connection timeout
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        print(client.server_info())
    except Exception:
        print("Unable to connect to the server.")

    return client[db]


def read_mongo(mongo_uri, db, collection, query={}, no_id=True):
    """ Read from Mongo and Store into DataFrame """
    # Connect to MongoDB
    db = _connect_mongo(mongo_uri, db)
    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)
    # Expand the cursor and construct the DataFrame
    df = pd.DataFrame(list(cursor))
    # Delete the _id
    if no_id:
        del df['_id']
    return df


def save_mongo(mongo_uri, db, collection, pred_pd):
    """ Save to Mongo and Store prediction """
    # Connect to MongoDB
    db = _connect_mongo(mongo_uri, db)
    # Make a query to the specific DB and Collection
    ins_res = db[collection].insert_many(pred_pd.to_dict('records'), ordered=0)
    print("total inserts", len(ins_res.inserted_ids))


