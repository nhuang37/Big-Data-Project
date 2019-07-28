#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# Imports
from pyspark.sql.functions import explode
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import RankingMetrics

def main(spark, df_test, model_file):

    # import model
    model = PipelineModel.load(model_file)
    print("imported model")
    
    # import test data
    test = spark.read.parquet(df_test)
    print("imported test data")
    
    # predict on test data
    testdf = model.transform(test)
    testdf = testdf.select([c for c in testdf.columns if c in ["user_index", "count", "track_index"]])

    # make labels
    testdf.createOrReplaceTempView('testdf')
    Labels = spark.sql('SELECT user_index, collect_list(track_index) AS label FROM testdf GROUP BY user_index')
    Labels.createOrReplaceTempView('Labels')
    print("created ground truth labels")

    # generate top 500 track recommendations for each user in validation set
    user_subset = testdf.select("user_index").distinct()
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

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName("baseline_test").getOrCreate()

    # Get the filename from the command line
    df_test = sys.argv[1]
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, df_test, model_file)
