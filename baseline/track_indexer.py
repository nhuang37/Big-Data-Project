#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# Imports
from pyspark.ml.feature import StringIndexer
from pyspark.ml.pipeline import Pipeline, PipelineModel

def main(spark, df, model_file):

    # import metadata
    df = spark.read.parquet(df)
    print("imported meta data")

    # make indexer on all the tracks in metadata
    indexer1 = StringIndexer(inputCol = "track_id", outputCol = "track_index", handleInvalid = "skip")
    pipeline = Pipeline(stages=[indexer1])
    model = pipeline.fit(df)
    print("mapped user_index")

    # output the model indexer
    model.write().overwrite().save(model_file)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName("track_indexer").getOrCreate()

    # Get the filename from the command line
    df = sys.argv[1]
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, df, model_file)
