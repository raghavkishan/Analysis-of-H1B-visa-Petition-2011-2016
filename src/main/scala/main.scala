import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{Binarizer, RFormula, VectorIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object main {
  def main(args: Array[String]): Unit = {

    var inputDataFile = "h1bMainData.csv"

    val sparkSession = SparkSession.builder().appName("randomForest").master("local[*]").getOrCreate()

    val h1bDatasetOne = sparkSession.read.format("csv").option("header", true).option("inferschema", true).load(inputDataFile)

    println("Before count:" + h1bDatasetOne.count())

    //removing rows that have no value for "CASE_STATUS", "SOC_NAME", "EMPLOYER_NAME
    val h1bDataset =  h1bDatasetOne.filter(col("CASE_STATUS").notEqual("NA")
      && col("SOC_NAME").notEqual("NA")
      && col("EMPLOYER_NAME").notEqual("NA"))

    println("After count:" + h1bDataset.count())

    // R formula
    val h1bFormula = new RFormula().setFormula("CASE_STATUS ~ FULL_TIME_POSITION + PREVAILING_WAGE")

    val fittedRF = h1bFormula.fit(h1bDataset)
    val preparedDF = fittedRF.transform(h1bDataset)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(preparedDF)

    //preparedDF.show()
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = preparedDF.randomSplit(Array(0.7, 0.3))

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")




    //Logistic Regression
    val logisticRegressor = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    // Chain indexers and forest in a Pipeline.
    val lrPipeline = new Pipeline().setStages(Array(featureIndexer, logisticRegressor))

    // Train model. This also runs the indexer.
    val lrModel = lrPipeline.fit(trainingData)

    // Make predictions.
    val lrpredictions = lrModel.transform(testData)

    val binarizer: Binarizer = new Binarizer()
      .setInputCol("prediction")
      .setOutputCol("binarized_prediction")
      .setThreshold(0.5)

    val predictionBinary = binarizer.transform(lrpredictions)

    // Select example rows to display.
    //predictions.select("prediction", "label", "features").show(5)

    val lrAccuracy = evaluator.evaluate(predictionBinary)
    println("*******************************************************************************************************************************************")
    println("*******************************************************************************************************************************************")
    println("Logistic Accuracy = " + lrAccuracy * 100)
    println("Logistic Test Error = " + (1.0 - lrAccuracy) * 100)
    println("*******************************************************************************************************************************************")
    println("*******************************************************************************************************************************************")

    //Decision Trees
    val DecisionTreeClassifier = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    // Chain indexers and forest in a Pipeline.
    val drPipeline = new Pipeline().setStages(Array(featureIndexer, DecisionTreeClassifier))

    // Train model. This also runs the indexer.
    val drModel = drPipeline.fit(trainingData)

    // Make predictions.
    val drPredictions = drModel.transform(testData)

    // Select example rows to display.
    //drPredictions.select("prediction", "label", "features").show(5)

    val draccuracy = evaluator.evaluate(drPredictions)
    println("*******************************************************************************************************************************************")
    println("*******************************************************************************************************************************************")
    println("Decision Tree Accuracy = "+draccuracy * 100)
    println("Decision Tree Test Error = " + (1.0 - draccuracy) * 100)
    println("*******************************************************************************************************************************************")
    println("*******************************************************************************************************************************************")



    //Random Forest
    val RandomForestClassifier = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    // Chain indexers and forest in a Pipeline.
    val rfpipeline = new Pipeline().setStages(Array(featureIndexer, RandomForestClassifier))

    // Train model. This also runs the indexer.
    val rfModel = rfpipeline.fit(trainingData)

    // Make predictions.
    val rfPredictions = rfModel.transform(testData)

    //rfPredictions.select("prediction", "label", "features").show(5)

    val rfAccuracy = evaluator.evaluate(rfPredictions)
    println("*******************************************************************************************************************************************")
    println("*******************************************************************************************************************************************")
    println("Random Forest Accuracy = "+rfAccuracy * 100)
    println("Random Forest Test Error = " + (1.0 - rfAccuracy) * 100)
    println("*******************************************************************************************************************************************")
    println("*******************************************************************************************************************************************")

    //Naive Bayes
    val NaiveBayesClassifier = new NaiveBayes().setLabelCol("label").setFeaturesCol("indexedFeatures")

    // Chain indexers and forest in a Pipeline.
    val nbpipeline = new Pipeline().setStages(Array(featureIndexer, NaiveBayesClassifier))

    // Train model. This also runs the indexer.
    val nbModel = nbpipeline.fit(trainingData)

    // Make predictions.
    val nbPredictions = nbModel.transform(testData)

    //nbPredictions.select("prediction", "label", "features").show(5)

    val nbAccuracy = evaluator.evaluate(nbPredictions)
    println("*******************************************************************************************************************************************")
    println("*******************************************************************************************************************************************")
    println("Naive Bayes Accuracy = "+nbAccuracy * 100)
    println("Naive Bayes Test Error = " + (1.0 - nbAccuracy) * 100)
    println("*******************************************************************************************************************************************")
    println("*******************************************************************************************************************************************")

    //One vs Rest classifier
    val ovrlogisticRegressor = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)
      .setTol(1E-6)
      .setFitIntercept(true)


    // instantiate the One Vs Rest Classifier.
    val ovr = new OneVsRest().setClassifier(ovrlogisticRegressor)

    // Chain indexers and forest in a Pipeline.
    val ovrPipeline = new Pipeline().setStages(Array(featureIndexer, ovr))

    // Train model. This also runs the indexer.
    val ovrModel = ovrPipeline.fit(trainingData)

    // Make predictions.
    val ovrpredictions = ovrModel.transform(testData)

    // Select example rows to display.
    //predictions.select("prediction", "label", "features").show(5)

    val ovrAccuracy = evaluator.evaluate(ovrpredictions)
    println("*******************************************************************************************************************************************")
    println("*******************************************************************************************************************************************")
    println("ovr Accuracy = " + ovrAccuracy * 100)
    println("ovr Test Error = " + (1.0 - ovrAccuracy) * 100)
    println("*******************************************************************************************************************************************")
    println("*******************************************************************************************************************************************")

  }
}
