#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# Imports
from pyspark.ml.pipeline import Pipeline, PipelineModel

def main(spark, df_metadata, model_file, output_file):

    # import model
    model = PipelineModel.load(model_file)
    print("imported model")
    
<<<<<<< HEAD
    # import test datat
=======
    # import metadata
>>>>>>> 5eb0a5d49b552c5f53e75dc8e349f6775e44ee7a
    df = spark.read.parquet(df_metadata)
    print("imported metadata")

    # transform metadata to get track_index
    metadf = model.transform(df)
    print("mapped track_index")

    # output a parquet file
    metadf.write.parquet(output_file)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName("metadata_transform").getOrCreate()

    # Get the filename from the command line
    df_metadata = sys.argv[1]
    model_file = sys.argv[2]
    output_file = sys.argv[3]

    # Call our main routine
    main(spark, df_metadata, model_file, output_file)
