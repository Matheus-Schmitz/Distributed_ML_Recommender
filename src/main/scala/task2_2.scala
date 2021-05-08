// DSCI 553 | Foundations and Applications of Data Mining
// Homework 3
// Matheus Schmitz
// USC ID: 5039286453

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import scala.collection.immutable
import scala.collection.mutable
import java.io._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import scala.util.control.NonFatal
import ml.dmlc.xgboost4j.scala.XGBoost
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.LabeledPoint


object task2_2 {
  implicit val formats: DefaultFormats.type = DefaultFormats

  case class Bizz_Json(business_id: String, name: String, neighborhood: String, address: String, city: String, state: String, postal_code: String,
                       latitude: Option[Float], longitude: Option[Float], stars: Float, review_count: Int, is_open: Int, attributes: Any, categories: String, hours: Any)


  case class User_Json(user_id: String, name: String, review_count: Int, yelping_since: String, friends: Any, useful: Int, funny: Int,
     cool: Int, fans: Int, elite: Any, average_stars: Float, compliment_hot: Int, compliment_more: Int, compliment_profile: Int, compliment_cute: Int, compliment_list: Int,
     compliment_note: Int, compliment_plain: Int, compliment_cool: Int, compliment_funny: Int, compliment_writer: Int, compliment_photos: Int)

  def get_features(row: ((Int, Int), Float),
                   user_features: scala.collection.Map[Int, (Float, Float, Float, Float, Float, Float)],
                   bizz_features: scala.collection.Map[Int, (Float, Float, Float, Float, String)]): List[Float] = {
    val yelp_categories: immutable.List[String] = immutable.List("Active Life", "Arts & Entertainment", "Automotive", "Beauty & Spas", "Education", "Event Planning & Services", "Financial Services", "Food",
    "Health & Medical", "Home Services", "Hotels & Travel", "Local Flavor", "Local Services", "Mass Media", "Nightlife", "Pets", "Professional Services",
    "Public Services & Government", "Real Estate", "Religious Organizations", "Restaurants", "Shopping")

    var user_X: List[Float] = List.empty[Float]
    try {
      user_X = List(user_features(row._1._1)._1, user_features(row._1._1)._2, user_features(row._1._1)._3, user_features(row._1._1)._4, user_features(row._1._1)._5, user_features(row._1._1)._6)
          }
    catch { case NonFatal(t) =>
      user_X = List(-1.toFloat, -1.toFloat, -1.toFloat, -1.toFloat, -1.toFloat, -1.toFloat)
            }

    var bizz_X: List[Float] = List.empty[Float]
    try {
      bizz_X = List(bizz_features(row._1._2)._1, bizz_features(row._1._2)._2, bizz_features(row._1._2)._3, bizz_features(row._1._2)._4)
          }
    catch { case NonFatal(t) =>
      bizz_X = List(-1.toFloat, -1.toFloat, -1.toFloat, -1.toFloat)
            }

    var categories_X: List[Float] = List.empty[Float]
    try {
      for (cat <- yelp_categories){
        if (bizz_features(row._1._2)._5 contains cat) {
          categories_X = categories_X :+ 1.0.toFloat}
        else {
          categories_X = categories_X :+ 0.0.toFloat}
      }
    }
    catch {
      case NonFatal(t) =>
        var categories_X: List[Float] = List.empty[Float]
        for (cat <- yelp_categories) {
          categories_X = categories_X :+ -1.toFloat
        }
    }

    val X: List[Float] = user_X ::: bizz_X ::: categories_X
  //    println("X: ", X)
    return X

}
  def classBizz(line: Any): Bizz_Json = {
    val splitter = line + ""
    parse(splitter).extract[Bizz_Json]
  }

  def classUser(line: Any): User_Json = {
    val splitter = line + ""
    parse(splitter).extract[User_Json]
  }

  def main(args: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()

    // Get user inputs
    val folder_path = args(0)
    val test_file_name = args(1)
    val output_file_name = args(2)

    // Initialize Spark with the 4 GB memory parameters from HW1
    val config = new SparkConf().setMaster("local[*]").setAppName("task2_2").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")//.set("spark.testing.memory", "471859200")
    val sc = SparkContext.getOrCreate(config)
    sc.setLogLevel("ERROR")

    // Read the CSV skipping its header, and reshape it as ((user, bizz), rating)
    val trainRDDwithHeader = sc.textFile(folder_path+"yelp_train.csv")
    val trainHeader = trainRDDwithHeader.first()
    val trainRDD = trainRDDwithHeader.filter(row => row != trainHeader).map(row => row.split(',')).map(row => ((row(0), row(1)), row(2).toFloat)).persist()
    val validRDDwithHeader = sc.textFile(test_file_name)
    val validHeader = validRDDwithHeader.first()
    val validRDD = validRDDwithHeader.filter(row => row != validHeader).map(row => row.split(',')).map(row => ((row(0), row(1)), 0.0.toFloat)).persist() // row(2).toFloat

    // Merge RDDs to get all IDs
    val mergedRDD = (trainRDD ++ validRDD)

    // Get distinct users and businesses (over train and valid datasets)
    val distinct_user = mergedRDD.map(row => row._1._1).distinct().sortBy(user => user).collect()
    val distinct_bizz = mergedRDD.map(row => row._1._2).distinct().sortBy(bizz => bizz).collect()

    // Convert names to IDs (to optimize memory usage when holding the values)
    val user_to_encoding = distinct_user.map(user => user -> distinct_user.indexOf(user)).toMap
    val encoding_to_user = distinct_user.map(user => distinct_user.indexOf(user) -> user).toMap
    val bizz_to_encoding = distinct_bizz.map(bizz => bizz -> distinct_bizz.indexOf(bizz)).toMap
    val encoding_to_bizz = distinct_bizz.map(bizz => distinct_bizz.indexOf(bizz) -> bizz).toMap

    // Use the IDs to encode the RDD, which reduces memory requirements when holding itemsets, and keep the shape as ((user, bizz), rating)
    val trainRDD_enc = trainRDD.map(x => ((user_to_encoding(x._1._1), bizz_to_encoding(x._1._2)), x._2)).persist()
    val validRDD_enc = validRDD.map(x => ((user_to_encoding(x._1._1), bizz_to_encoding(x._1._2)), x._2)).persist()

    // Read the user and business jsons, and load the features
    val user_features = sc.textFile(folder_path+"user.json").map(classUser).filter(row => distinct_user contains row.user_id).map(row => (user_to_encoding(row.user_id), (row.average_stars, row.review_count.toFloat, row.yelping_since.substring(2,4).toFloat, row.useful.toFloat, row.funny.toFloat, row.cool.toFloat))).collectAsMap()
    val bizz_features = sc.textFile(folder_path+"business.json").map(classBizz).filter(row => distinct_bizz contains row.business_id).map(row => (bizz_to_encoding(row.business_id), (row.stars, row.review_count.toFloat, row.latitude.getOrElse(-1.toFloat), row.longitude.getOrElse(-1.toFloat), row.categories))).collectAsMap()
        
    // Get train data features
    val data_train = trainRDD_enc.map(row => (get_features(row, user_features, bizz_features), row._2)).map(x => new LabeledPoint(label = x._2, values = x._1.toArray[Float], indices = null)).collect().toIterator
    val dmat_train = new DMatrix(data_train)

    // Get test data features
    val data_test = validRDD_enc.map(row => (get_features(row, user_features, bizz_features), row._2)).map(x => new LabeledPoint(label = x._2, values = x._1.toArray[Float], indices = null)).collect().toIterator
    val dmat_test = new DMatrix(data_test)

    // Define hyperparameters
    val xgb_params = Map(
      "num_workers" -> -1,
      "learning_rate" -> 0.1,
      "booster" -> "gbtree",
      "eval_metric" -> "rmse",
      "updater" -> "grow_colmaker",
      "min_child_weight" -> 0,
      "min_split_loss" -> 0,
      "subsample" -> 1,
      "colsample_bytree" -> 1,
      "max_depth" -> 4,
      "reg_lambda" -> 0.0,
      "reg_alpha" -> 0.0)

    // Report RMSE
    val watches: Map[String, DMatrix] = Map(
      "train" -> dmat_train,
      "test" -> dmat_test)

    // Train model
    val model = XGBoost.train(dtrain = dmat_train, params = xgb_params, round = 200) //, watches)

    // Predict
    val predictions = model.predict(dmat_test)

    // Bind predictions to [1, 5]
    val predictions_capped: mutable.ListBuffer[Float] = mutable.ListBuffer[Float]()
    for (pred <- predictions){
      predictions_capped += pred(0).max(1.0.toFloat).min(5.0.toFloat)
    }

    // Write results
    val txtoutputter = new PrintWriter(new File(output_file_name))
    txtoutputter.write("user_id, business_id, prediction")
    var n = 0
    for (row <- validRDD_enc.collect()) {
      txtoutputter.write("\n" + encoding_to_user(row._1._1) + "," + encoding_to_bizz(row._1._2) + "," + predictions_capped(n))
      n += 1
    }
    txtoutputter.close()

    // Close spark context
    sc.stop()

    // Measure the total time taken and report it
    val total_time = System.currentTimeMillis() - start_time
    val time_elapsed = (total_time).toFloat / 1000.toFloat
    println("Duration: " + time_elapsed)
  }
}
