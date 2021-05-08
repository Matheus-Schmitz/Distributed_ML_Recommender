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
import scala.util.Random


object task2_3 {
  implicit val formats: DefaultFormats.type = DefaultFormats

  case class Bizz_Json(business_id: String, name: String, neighborhood: String, address: String, city: String, state: String, postal_code: String,
                       latitude: Option[Float], longitude: Option[Float], stars: Float, review_count: Int, is_open: Int, attributes: Any, categories: String, hours: Any)


  case class User_Json(user_id: String, name: String, review_count: Int, yelping_since: String, friends: Any, useful: Int, funny: Int,
                       cool: Int, fans: Int, elite: Any, average_stars: Float, compliment_hot: Int, compliment_more: Int, compliment_profile: Int, compliment_cute: Int, compliment_list: Int,
                       compliment_note: Int, compliment_plain: Int, compliment_cool: Int, compliment_funny: Int, compliment_writer: Int, compliment_photos: Int)

  def get_features_train(row: ((Int, Int), Float),
                   user_features: scala.collection.Map[Int, (Float, Float, Float, Float, Float, Float)],
                   bizz_features: scala.collection.Map[Int, (Float, Float, Float, Float, String)],
                   item_CF_feature_train: scala.collection.Map[(Int, Int), Float] ): List[Float] = {
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

    var item_CF_X: List[Float] = List.empty[Float]
    try {         //Add some noise to the train CF to stop XGBoost from overfitting
      item_CF_X = List(item_CF_feature_train(row._1._1, row._1._2) * (1 + Random.nextGaussian()*0.33).toFloat)
    }
    catch { case NonFatal(t) =>
      item_CF_X = List(-1.toFloat)
    }

    val X: List[Float] = user_X ::: bizz_X ::: categories_X ::: item_CF_X
    //    println("X: ", X)
    return X
  }

  def get_features_test(row: ((Int, Int), Float),
                         user_features: scala.collection.Map[Int, (Float, Float, Float, Float, Float, Float)],
                         bizz_features: scala.collection.Map[Int, (Float, Float, Float, Float, String)],
                         item_CF_feature_test: scala.collection.Map[(Int, Int), Float] ): List[Float] = {
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

    var item_CF_X: List[Float] = List.empty[Float]
    try {
      item_CF_X = List(item_CF_feature_test(row._1._1, row._1._2))
    }
    catch { case NonFatal(t) =>
      item_CF_X = List(-1.toFloat)
    }

    val X: List[Float] = user_X ::: bizz_X ::: categories_X ::: item_CF_X
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

  def item_based_CF(user: Int,
                    bizz: Int,
                    user_avg_rating: scala.collection.Map[Int, Float],
                    bizz_avg_rating: scala.collection.Map[Int, Float],
                    user_bizz_rating_dict: scala.collection.Map[Int, Iterable[(Int, Float)]],
                    bizz_user_rating_dict: scala.collection.Map[Int, Map[Int, Float]]): ((Int, Int), Float) = {

    /// Ensure no errors in case a user and / or business doesn 't have an average rating score ### #If both user and business have missing ratings, return the best guess, aka 3
    if (!user_avg_rating.contains(user) && (!bizz_avg_rating.contains(bizz))) {
      return ((user, bizz), (3.75).toFloat) }

    // If only the business has a missing value, we still cannot calculate similarity, so return the average for the associated user
    else if (!bizz_avg_rating.contains(bizz)) {
      return ((user, bizz), user_avg_rating(user))}

    // If only the user has a missing value, we still cannot calculate similarity, so return the average for the associated business
    else if (!user_avg_rating.contains(user)) {
      return ((user, bizz), bizz_avg_rating(bizz))}

    // If both user and business have ratings, proceed to calculating similarities
    var similarities: mutable.ListBuffer[(Int, Float)] = mutable.ListBuffer.empty[(Int, Float)]

    // For each business rated by the current user, calculate the similarity between the current business and the comparison business
    val bizz_rating_dict = user_bizz_rating_dict(user)
    for (encoding <- 0 until bizz_rating_dict.size) {
      var pearson_corr = item_item_similarity(bizz, bizz_rating_dict.slice(encoding, encoding + 1).toList.head._1, bizz_user_rating_dict)
      // Skip similarities of 0 to gain performenace
      if (pearson_corr != 0) {
        similarities.append((encoding, pearson_corr))
      }
    }
    // Calculate the person correlation to make a weighted prediction
    var N = 0.toFloat
    var D = 0.toFloat

    for ((encoding, pearson_corr) <- similarities) {
      val bizz_rating_tuple = bizz_rating_dict.slice(encoding, encoding + 1).toList.head
      val business = bizz_rating_tuple._1
      val rating = bizz_rating_tuple._2
      val business_avg_rating = bizz_avg_rating(business)
      N += (rating - business_avg_rating) * pearson_corr
      D += pearson_corr.abs
    }

    if (N != 0 ) {
      val prediction = (bizz_avg_rating(bizz) + N / D).toFloat
      return ((user, bizz), prediction)
    }

    else {
      val prediction = 3.75.toFloat
      return ((user, bizz), prediction)
    }
  }

  def item_item_similarity(curr_bizz: Int,
                           comp_bizz: Int,
                           bizz_user_rating_dict: scala.collection.Map[Int, Map[Int, Float]]): Float = {

    // For each business get all pairs of user / rating
    val curr_bizz_ratings = bizz_user_rating_dict(curr_bizz)
    val comp_bizz_ratings = bizz_user_rating_dict(comp_bizz)

    // Get co - rated users (those who rated both businesses)
    val corated_users = curr_bizz_ratings.keys.toSet.intersect(comp_bizz_ratings.keys.toSet)

    // If there are no co - rated users, its impossible to calculate similarity, so return a guess
    if (corated_users.isEmpty) {
      return 0.5.toFloat
    }

    // Calculate the average rating given to the businesses by the co - rated users
    var curr_bizz_total = 0.toFloat
    var comp_bizz_total = 0.toFloat
    var count = 0.toFloat

    for (user <- corated_users) {
      curr_bizz_total += curr_bizz_ratings(user)
      comp_bizz_total += comp_bizz_ratings(user)
      count += 1
    }

    var curr_bizz_avg = curr_bizz_total / count
    var comp_bizz_avg = comp_bizz_total / count

    // Calculate the pearson correlation
    var curr_x_comp_total = 0.toFloat
    var curr_norm_square = 0.toFloat
    var comp_norm_square = 0.toFloat

    for (user <- corated_users) {
      curr_x_comp_total += ((curr_bizz_ratings(user) - curr_bizz_avg) * (comp_bizz_ratings(user) - comp_bizz_avg)).toFloat
      curr_norm_square += math.pow((curr_bizz_ratings(user) - curr_bizz_avg), 2).toFloat
      comp_norm_square += math.pow((comp_bizz_ratings(user) - comp_bizz_avg), 2).toFloat
    }

    // Get the Pearson Correlation (Of guess a correlation if we cannot calculate the correlation for a given pair)
    if (curr_x_comp_total != 0 ) {
      val pearson_corr = curr_x_comp_total / (math.pow(curr_norm_square, 0.5) * math.pow(comp_norm_square, 0.5)).toFloat
      return pearson_corr
    }
    else {
      val pearson_corr = 0.5.toFloat
      return pearson_corr}

  }

  def main(args: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()

    // Get user inputs
    val folder_path = args(0)
    val test_file_name = args(1)
    val output_file_name = args(2)

    // Initialize Spark with the 4 GB memory parameters from HW1
    val config = new SparkConf().setMaster("local[*]").setAppName("task2_2").set("spark.executor.memory", "6g").set("spark.driver.memory", "6g")//.set("spark.testing.memory", "471859200")
    val sc = SparkContext.getOrCreate(config)
    sc.setLogLevel("ERROR")

    // Read the CSV skipping its header, and reshape it as ((user, bizz), rating)
    val trainRDDwithHeader = sc.textFile(folder_path+"yelp_train.csv")
    val trainHeader = trainRDDwithHeader.first()
    val trainRDD = trainRDDwithHeader.filter(row => row != trainHeader).map(row => row.split(',')).map(row => ((row(0), row(1)), row(2).toFloat)).persist()
    val validRDDwithHeader = sc.textFile(test_file_name)
    val validHeader = validRDDwithHeader.first()
    val validRDD = validRDDwithHeader.filter(row => row != validHeader).map(row => row.split(',')).map(row => ((row(0), row(1)), 0.0.toFloat)).persist() // row(2)toFloat

    // Merge RDDs to get all IDs
    var mergedRDD = (trainRDD ++ validRDD)

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


    //--------------------------------//
    // Item-Item Similarity Features  //
    //--------------------------------//

    // Calculate average ratings
    val user_avg_rating = trainRDD_enc.map(x => (x._1._1, x._2)).groupByKey().map(row => (row._1, row._2.sum / row._2.size)).collectAsMap()
    val bizz_avg_rating = trainRDD_enc.map(x => (x._1._2, x._2)).groupByKey().map(row => (row._1, row._2.sum / row._2.size)).collectAsMap()

    // For each user/bizz, get a dict with all related bizz/user and the associated rating
    val user_bizz_rating_dict = trainRDD_enc.map(x => (x._1._1, (x._1._2, x._2))).groupByKey().collectAsMap()
    val bizz_user_rating_dict = trainRDD_enc.map(x => (x._1._2, (x._1._1, x._2))).groupByKey().map(x => (x._1, x._2.toMap)).collectAsMap()

    // Make predictions
    val item_CF_feature_train = trainRDD_enc.map(row => item_based_CF(row._1._1,
                                                                      row._1._2,
                                                                      user_avg_rating,
                                                                      bizz_avg_rating,
                                                                      user_bizz_rating_dict,
                                                                      bizz_user_rating_dict)).collectAsMap()

    val item_CF_feature_test = validRDD_enc.map(row => item_based_CF(row._1._1,
                                                                    row._1._2,
                                                                    user_avg_rating,
                                                                    bizz_avg_rating,
                                                                    user_bizz_rating_dict,
                                                                    bizz_user_rating_dict)).collectAsMap()

    //---------------------------------------//
    // User/Business Complimentary Features  //
    //---------------------------------------//

    // Read the user and business jsons, and load the features
    val user_features = sc.textFile(folder_path+"user.json").map(classUser).filter(row => distinct_user contains row.user_id).map(row => (user_to_encoding(row.user_id), (row.average_stars, row.review_count.toFloat, row.yelping_since.substring(2,4).toFloat, row.useful.toFloat, row.funny.toFloat, row.cool.toFloat))).collectAsMap()
    val bizz_features = sc.textFile(folder_path+"business.json").map(classBizz).filter(row => distinct_bizz contains row.business_id).map(row => (bizz_to_encoding(row.business_id), (row.stars, row.review_count.toFloat, row.latitude.getOrElse(-1.toFloat), row.longitude.getOrElse(-1.toFloat), row.categories))).collectAsMap()


    //-----------------------------------------//
    // Feature Augmentation Hybrid Recommender //
    //-----------------------------------------//

    // Get train data features
    //val data_train = trainRDD_enc.map(row => (get_features_train(row, user_features, bizz_features, item_CF_feature_train), row._2)).map(x => new LabeledPoint(label = x._2, values = x._1.toArray[Float], indices = null)).collect().toIterator
    val dmat_train = new DMatrix(trainRDD_enc.map(row => (get_features_train(row, user_features, bizz_features, item_CF_feature_train), row._2)).map(x => new LabeledPoint(label = x._2, values = x._1.toArray[Float], indices = null)).collect().toIterator)

    // Get test data features
    //val data_test = validRDD_enc.map(row => (get_features_test(row, user_features, bizz_features, item_CF_feature_test), row._2)).map(x => new LabeledPoint(label = x._2, values = x._1.toArray[Float], indices = null)).collect().toIterator
    val dmat_test = new DMatrix(validRDD_enc.map(row => (get_features_test(row, user_features, bizz_features, item_CF_feature_test), row._2)).map(x => new LabeledPoint(label = x._2, values = x._1.toArray[Float], indices = null)).collect().toIterator)


    // Define hyperparameters
    val xgb_params = Map(
      "num_workers" -> -1,
      "learning_rate" -> 0.1,
      "booster" -> "gbtree",
      "eval_metric" -> "rmse",
      "updater" -> "grow_colmaker",
      "min_child_weight" -> 0,
      "min_split_loss" -> 0,
      "subsample" -> 0.5,
      "colsample_bytree" -> 0.5,
      "max_depth" -> 4,
      "reg_lambda" -> 0.0,
      "reg_alpha" -> 0.0)

    // Report RMSE
    val watches: Map[String, DMatrix] = Map(
      "train" -> dmat_train,
      "test" -> dmat_test)

    // Train model
    val model = XGBoost.train(dtrain = dmat_train, params = xgb_params, round = 100) //, watches)

    // Predict
    //val predictions = model.predict(dmat_test)

    // Bind predictions to [1, 5]
    val predictions_capped: mutable.ListBuffer[Float] = mutable.ListBuffer[Float]()
    for (pred <- model.predict(dmat_test)){
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
