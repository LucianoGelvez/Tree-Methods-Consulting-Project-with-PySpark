from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

# Create Spark session
spark = SparkSession.builder.appName('footPet').getOrCreate()

# Load data
data = spark.read.csv('dog_food.csv', inferSchema=True, header=True)

# Print schema to ensure correct data loading
data.printSchema()

# Display first two rows of the dataset
print(data.head(2))

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=['A', 'B', 'C', 'D'], outputCol='features')
output = assembler.transform(data)

# Print schema to ensure features column is added
output.printSchema()

# Select final data for model training
final_data = output.select('features', 'Spoiled')

# Display final data
final_data.show()

# Initialize Random Forest Classifier
rfc = RandomForestClassifier(labelCol='Spoiled', featuresCol='features')

# Train the model
rfc_model = rfc.fit(final_data)

# Display feature importances
print(rfc_model.featureImportances)