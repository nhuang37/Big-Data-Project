#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# Imports
import itertools
from pyspark.sql.functions import explode
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import RankingMetrics

def main(spark, df_train, df_val, model_file):

    # import train data
    train = spark.read.parquet(df_train)
    print("imported train data")
    
    # import validation data
    val = spark.read.parquet(df_val)
    print("imported validation data")
    
    # index users and tracks
    indexer1 = StringIndexer(inputCol = "user_id", outputCol = "user_index", handleInvalid = "skip") #skip null values

    # grid search
    bestModel = None
    bestValidationMAP = -1
    best_rank, best_regparam, best_alpha = None, None, None
    list_regParam = [0.05]
    list_rank = [10,20,50] 
    list_alpha = [1,15]

    # drop count threshold
    #drop_count = [1,2,3]
    
    # select records with count > 1
    train = train.filter(train["count"] > 1)
    val = val.filter(val["count"] > 1)
    print("kept records with count > 1")
    
    # Build the recommendation model using ALS on the train data
    for reg, rank, alpha in itertools.product(list_regParam, list_rank, list_alpha):

        als = ALS(seed = 1, rank = rank, regParam = reg, alpha = alpha, userCol = "user_index", itemCol = "track_index", ratingCol = "count", implicitPrefs = True)
    
        # create pipeline
        pipeline = Pipeline(stages=[indexer1,als])
        model = pipeline.fit(train)
        print("trained model with reg = %s, rank = %s, alpha = %s" %(reg, rank, alpha))
        
        # predict on validation data and indexed users
        val_indexed = model.transform(val)
        val_indexed = val_indexed.select([c for c in val_indexed.columns if c in ["user_index", "count", "track_index"]])
        print("indexed users")

        # make labels
        val_indexed.createOrReplaceTempView('val_indexed')
        Labels = spark.sql('SELECT user_index, collect_list(track_index) AS label FROM val_indexed GROUP BY user_index')
        Labels.createOrReplaceTempView('Labels')
        print("created ground truth labels")

        # generate top 500 track recommendations for each user in validation set
        user_subset = val_indexed.select("user_index").distinct()
        userRecs = model.stages[-1].recommendForUserSubset(user_subset,500)
        userRecs.createOrReplaceTempView("userRecs")
        print("made user recommendations")

        # explode recommendations in long format
        Recs = (userRecs.select("user_index", explode("recommendations").alias("pred")).select("user_index", "pred.*"))
        Recs.createOrReplaceTempView("Recs")

        # make predictions
        Preds = spark.sql('SELECT user_index, collect_list(track_index) AS prediction FROM Recs GROUP BY user_index')
        Preds.createOrReplaceTempView("Preds")

        # make label pairs
        Preds_labels = spark.sql('SELECT Preds.prediction AS prediction, Labels.label as label FROM Preds INNER JOIN Labels ON Preds.user_index = Labels.user_index')
        print("inner join preds & labels")

        # calculate MAP
        MAPrecommendationsAndTruth = Preds_labels.select("prediction", "label")
        metrics = RankingMetrics(MAPrecommendationsAndTruth.rdd)
        MAP = metrics.meanAveragePrecision
        print("MAP = %s" % MAP)

        # get best model
        if MAP > bestValidationMAP:
            bestModel = model
            bestValidationMAP = MAP
            best_rank, best_regparam, best_alpha = rank, reg, alpha

    # save best model and params
    pip_model = bestModel
    pip_model.write().overwrite().save(model_file)
    print("Best model saved with reg = %s, rank = %s, alpha = %s, MAP = %s" %(best_regparam, best_rank, best_alpha, bestValidationMAP))

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName("extension1_dropcount_train").getOrCreate()

    # Get the data file from the command line
    df_train = sys.argv[1]
    df_val = sys.argv[2] 
    model_file = sys.argv[3]

    # Call our main routine
    main(spark, df_train, df_val, model_file)
