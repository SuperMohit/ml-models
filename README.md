# xgboost-model

This repo consists of xgboost model to make a prediction on Time Series Data stored in MongoDB. 
The Training and Test data is read from the MongoDB collections and converted into Pandas data frame. 
There are multiple collections from where data can be read and analysed. The Trained model is also then "pickled" and saved into the MongoDB collection.
The saved model can be further used for making prediction with other set of inputs. 


This repository would be exposed via a Rapid Api GRPC. Which would take the relevent inputs for instance the training and test data location i.e Namespace in MongoDB.

 



