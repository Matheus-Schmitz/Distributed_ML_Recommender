// DSCI 553 | Foundations and Applications of Data Mining
// Homework 3
// Matheus Schmitz
// USC ID: 5039286453

// spark-submit --class task1 hw3.jar

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import scala.collection.mutable
import java.io._

object task1 {

  // Specify hyperparameters for hashing
  val n_hash = 80
  val n_band = 40
  val n_rows = n_hash / n_band

  def hash_function(user: Int, hash_idx: Int, n_user: Int): Int = {
    // Some potential hash functions could be of type:
    // f(x) = (ax + b) % m or f(x) = ((ax + b) % p) % m
    return (user * hash_idx) % n_user
  }


  def get_signatures(bizz_id: String, users_rated: Iterable[Int], n_user: Int): (String, mutable.MutableList[Float]) = {
    // Initializ the signature matrix
    val signature_matrix = mutable.MutableList.fill(n_hash)(Float.PositiveInfinity)
    // Loop through each business rated by the user
    for (user <- users_rated) {
      // Generated the desired number of hash functions
      for (hash_func <- 0 until n_hash) {
        // Find the "converted index" of the current hash_func
        var conv_idx = hash_function(user, hash_func, n_user)
        //If the new index is lower than the previous, update the signature matrix
        if (conv_idx < signature_matrix(hash_func)) {
          signature_matrix(hash_func) = conv_idx
        }
      }
    }
    // Then return the signature matrix for a given user
    //signature_matrix.map(_.toInt)
    return (bizz_id, signature_matrix)
  }

  def get_bands(bizz_signature_tuple: (String, Iterable[Float])): List[((Int, Tuple1[Iterable[Float]]), String)] = {
    val bands = mutable.ListBuffer.empty[((Int, Tuple1[Iterable[Float]]), String)]
    // Loop through the bands
    for (band_num <- 0 until n_band) {
      // And append the signature index at the key corresponding to the current band
      bands.append(Tuple2(Tuple2(band_num, Tuple1(bizz_signature_tuple._2.slice(band_num * n_rows, (band_num + 1) * n_rows))), bizz_signature_tuple._1))
    }
    return bands.toList
  }

  def get_combinations(bizz: ((Int, Tuple1[Iterable[Float]]), Iterable[String])): Iterator[Set[String]] = {
    return bizz._2.toSet.subsets(2)
  }


  def jaccard_similarity(pair: Set[String], bizz_x_users_dict: scala.collection.Map[String, Set[Int]]): ((String, String), Double) = {
    // Sort the pairs to ensure properly ordered output
    val sorted_pairs = pair.toArray.sorted

    // Get the users who rated each of the businesses
    val bizz1 = bizz_x_users_dict(sorted_pairs(0)).toSet
    val bizz2 = bizz_x_users_dict(sorted_pairs(1)).toSet

    // Get intersection and unions
    val intersection = bizz1.intersect(bizz2)
    val union = bizz1.union(bizz2)

    // Calculate Jaccard Similarity
    var similarity = intersection.size.toDouble / union.size.toDouble

    // Return a key-value pair with the businesses and their similarity
    return ((sorted_pairs(0), sorted_pairs(1)), similarity)
  }


  def main(args: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()

    val input_file_path = args(0)
    val output_file_path = args(1)
    //val input_file_path = "data/yelp_train.csv"
    //val output_file_path = "scala1a.csv"

    // Initialize Spark with the 4 GB memory parameters from HW1
    val config = new SparkConf().setMaster("local[*]").setAppName("task1").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")//.set("spark.testing.memory", "471859200")
    val sc = SparkContext.getOrCreate(config)

    // Specify hyperparameters for hashing
    //val n_hash = sc.broadcast(80)
    //val n_band = sc.broadcast(40)
    //val n_rows = sc.broadcast(n_hash.value/n_band.value)

    // Read the CSV skipping its header
    val csvRDDwithHeader = sc.textFile(input_file_path)
    val csvHeader = csvRDDwithHeader.first()
    val csvRDD = csvRDDwithHeader.filter(row => row != csvHeader).map(row => row.split(',')).persist()

    // Get distinct users
    val userRDD = csvRDD.map(row => row(0)).distinct().sortBy(user => user).collect()

    // Get number of users
    val n_user = userRDD.length

    // Create unique IDs for each user
    val user_id_dict = userRDD.toList.map(user => user -> userRDD.indexOf(user)).toMap

    // Input matrix (Key = Business, Value = List of Users that rated the business)
    val bizz_x_users = csvRDD.map(row => (row(1), user_id_dict(row(0)))).groupByKey().sortBy(row => row._1)
    val bizz_x_users_dict = bizz_x_users.mapValues(users_list => users_list.toSet).collectAsMap()

    // Generate signatures
    val signatures = bizz_x_users.map(row => get_signatures(row._1, row._2, n_user))

    // Then perform LSH, then filter to keep only bands which have more than 1 business mapped to them
    val sig_bands = signatures.flatMap(sig => get_bands(sig)).groupByKey().filter(sig => sig._2.size > 1)

    // Generate candidate pairs for comparison (based on business that mapped to the same band)
    val candidates = sig_bands.flatMap(row => get_combinations(row)).distinct()

    // Calculate similarity between pairs, keep those with similarity over 0.5
    val similarity_dict = candidates.map(pair => jaccard_similarity(pair, bizz_x_users_dict)).filter(sim => sim._2 >= 0.5).sortBy(row => (row._1._1, row._1._2)).collect()

    // Write results
    val txtoutputter = new PrintWriter(new File(output_file_path))
    txtoutputter.write("business_id_1, business_id_2, similarity")
    for (item <- similarity_dict) {
      txtoutputter.write("\n" + item._1._1 + "," + item._1._2 + "," + item._2)
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
