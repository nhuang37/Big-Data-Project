#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# Imports
from pyspark.ml.pipeline import Pipeline, PipelineModel

def main(spark, df_user, model_file, output_file):

    # import model
    model = PipelineModel.load(model_file)
    print("imported model")
    
    # import user data
    df = spark.read.parquet(df_user)
    print("imported user data")

    # transform metadata to get track_index
    userdf = model.stages[0].transform(df)
    print("mapped user_index")

    # output a parquet file
    userdf.write.parquet(output_file)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName("user_transform").getOrCreate()

    # Get the filename from the command line
    df_user = sys.argv[1]
    model_file = sys.argv[2]
    output_file = sys.argv[3]

    # Call our main routine
    main(spark, df_user, model_file, output_file)
