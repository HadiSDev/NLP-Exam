from NERDA.models import NERDA
from NERDA.datasets import get_dane_data, download_dane_data
import nltk
from sklearn.model_selection import train_test_split
import pandas as pd
import os

nltk.download('punkt')

hyperparameters: dict = {'epochs': 15,
                         'warmup_steps': 500,
                         'train_batch_size': 32,
                         'learning_rate': 0.0001}
validation_batch_size = 16


def RunDaneNer(train, val):
    modelDaneNer = NERDA(dataset_training=train,
                         transformer='Maltehb/danish-bert-botxo-ner-dane',
                         dataset_validation=val,
                         validation_batch_size=validation_batch_size,
                         hyperparameters=hyperparameters)
    modelDaneNer.train()
    resDaneNer = modelDaneNer.evaluate_performance(get_dane_data('test'))
    resDaneNer.to_csv(r"results/resDaneNer.csv")


def RunDane(train, val):
    modelDane = NERDA(dataset_training=train,
                      transformer='Maltehb/danish-bert-botxo',
                      dataset_validation=val,
                      validation_batch_size=validation_batch_size,
                      hyperparameters=hyperparameters)
    modelDane.train()
    resDane = modelDane.evaluate_performance(get_dane_data('test'))
    resDane.to_csv(r"results/resDane.csv")


def RunSwedishNer(train, val):
    modelSwedishNer = NERDA(dataset_training=train,
                            transformer='KB/bert-base-swedish-cased-ner',
                            dataset_validation=val,
                            validation_batch_size=validation_batch_size,
                            hyperparameters=hyperparameters)
    modelSwedishNer.train()
    resSwedishNer = modelSwedishNer.evaluate_performance(get_dane_data('test'))
    resSwedishNer.to_csv(r"results/resSwedishNer.csv")


def RunSwedish(train, val):
    modelSwedish = NERDA(dataset_training=train,
                         transformer='KB/bert-base-swedish-cased',
                         dataset_validation=val,
                         validation_batch_size=validation_batch_size,
                         hyperparameters=hyperparameters)
    modelSwedish.train()
    resSwedish = modelSwedish.evaluate_performance(get_dane_data('test'))
    resSwedish.to_csv(r"results/resSwedish.csv")


def RunNorweigan(train, val):
    modelNorweigan = NERDA(dataset_training=train,
                           transformer='NbAiLab/nb-bert-large',
                           dataset_validation=val,
                           validation_batch_size=validation_batch_size,
                           hyperparameters=hyperparameters)
    modelNorweigan.train()
    resNorweigan = modelNorweigan.evaluate_performance(get_dane_data('test'))
    resNorweigan.to_csv(r"results/resNorweigan.csv")


def RunFinnish(train, val):
    modelFinnish = NERDA(dataset_training=train,
                         transformer='TurkuNLP/bert-base-finnish-cased-v1',
                         dataset_validation=val,
                         validation_batch_size=validation_batch_size,
                         hyperparameters=hyperparameters)
    modelFinnish.train()
    resFinnish = modelFinnish.evaluate_performance(get_dane_data('test'))
    resFinnish.to_csv(r"results/resFinnish.csv")


def RunIcelandic(train, val):
    modelIceLandic = NERDA(dataset_training=train,
                           transformer='m3hrdadfi/icelandic-ner-bert',
                           dataset_validation=val,
                           validation_batch_size=validation_batch_size,
                           hyperparameters=hyperparameters)
    modelIceLandic.train()
    resIceLandic = modelIceLandic.evaluate_performance(get_dane_data('test'))
    resIceLandic.to_csv(r"results/resIceLandic.csv")


def RunMulti(train, val):
    modelMulti = NERDA(dataset_training=train,
                       dataset_validation=val,
                       validation_batch_size=validation_batch_size,
                       hyperparameters=hyperparameters)
    modelMulti.train()
    resMulti = modelMulti.evaluate_performance(get_dane_data('test'))
    resMulti.to_csv(r"results/resMulti.csv")


if __name__ == "__main__":

    try:
        os.mkdir("results")
    except OSError as error:
        print(error)

    download_dane_data()
    testdf = pd.DataFrame()
    testdf.to_csv(r"results/resTest.csv")
    data = get_dane_data('train')
    test = get_dane_data('test')
    test_n = len(test["sentences"])
    X_train, X_val, Y_train, Y_val = train_test_split(data["sentences"], data["tags"], train_size=0.8)

    dataTrain = {"sentences": X_train, "tags": Y_train}
    dataVal = {"sentences": X_val, "tags": Y_val}

    RunDaneNer(dataTrain, dataVal)

    RunSwedish(dataTrain, dataVal)

    RunSwedishNer(dataTrain, dataVal)

    RunNorweigan(dataTrain, dataVal)

    RunFinnish(dataTrain, dataVal)

    RunIcelandic(dataTrain, dataVal)

    RunMulti(dataTrain, dataVal)
