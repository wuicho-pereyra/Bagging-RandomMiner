package org.apache.spark.mllib.classification

import java.io.Serializable

import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * Distributed RandomMinwe Class.
 *
  * @param train: Traning set
  * @param numberEnsembles: Number of ensembles
  * @param porcentEnsembles: Porcent of each ensemble
  * @param dist: Type of distance, euclidean, manhattan or chebyshev
  * @author luispereyra
  */


class RandomMiner(train: RDD[LabeledPoint], numberEnsembles: Int, porcentEnsembles: Double, porcentMROs: Double, dist: Int) extends Serializable {

  var distanceType: String = ""
  var modelValues = new Array[(Double, RDD[(Int, Vector)])](numberEnsembles)
  var inc = 0
  var subdel = 0
  var topdel = 0
  var numIter = 0

  private def broadcastMROs(MROs: Array[(Int, Vector)], context: SparkContext) = context.broadcast(MROs)


  if (dist == 1) {
    distanceType = "Manhattan"
  } else if (dist == 2) {
    distanceType = "Euclidean"
  } else if (dist == 3) {
    distanceType = "Chebyshev"
  } else if (dist == 4) {
    distanceType = "Mahalanobis"
  }

  def MROsDistances[T](iter: Iterator[(Int, Vector)]): Iterator[Double] = {
    // Initialization
    var trainPartition = new ArrayBuffer[(Int, Vector)]

    //Join the train set
    while (iter.hasNext)
      trainPartition.append(iter.next)

    val size = trainPartition.size

    var result = new ArrayBuffer[Double]

    for (i <- 0 until size) {
      for (j <- i + 1 until size) {
        result.append(calculateDistance.apply(trainPartition(i)._2, trainPartition(j)._2, distanceType))
      }
    }
    result.iterator
  }


  def tipicality[T](umbral: Double, MROs: Broadcast[Array[(Int, Vector)]], iter: Iterator[(Int, Vector)]): Iterator[(Int, Double)] = {
    val size = MROs.value.length
    var testPartition = new ArrayBuffer[(Int, Vector)]
    var testTipically = new ArrayBuffer[(Int, Double)]

    while (iter.hasNext)
      testPartition.append(iter.next)

    val sizeTest = testPartition.size
    val comparitions = (math.log10(MROs.value.length) / math.log10(2)).toInt

    for (i <- 0 until sizeTest) {
      var closeDistance = -1.0
      //for (j <- 0 until comparitions) {
      for (j <- 0 until size) {
        if (closeDistance == -1.0) closeDistance = calculateDistance.apply(MROs.value(j)._2, testPartition(i)._2, distanceType)
        val distance = calculateDistance.apply(MROs.value(j)._2, testPartition(i)._2, distanceType)
        if (distance < closeDistance) closeDistance = distance
      }
      val typical = math.exp(-0.5 * ((closeDistance / umbral) * (closeDistance / umbral)))
      testTipically.append((testPartition(i)._1, typical))
    }
    testTipically.iterator
  }

  def tipicalityMahalanobis[T](umbral: Double, MROs: Broadcast[Array[(Int, Vector)]], iter: Iterator[(Int, Vector)], inverse: Matrix): Iterator[(Int, Double)] = {
    val size = MROs.value.length
    var testPartition = new ArrayBuffer[(Int, Vector)]
    var testTipically = new ArrayBuffer[(Int, Double)]

    while (iter.hasNext)
      testPartition.append(iter.next)

    val sizeTest = testPartition.size
    val comparitions = (math.log10(MROs.value.length) / math.log10(2)).toInt

    for (i <- 0 until sizeTest) {
      var closeDistance = -1.0
      for (j <- 0 until comparitions) {
      //for (j <- 0 until size) {
        if (closeDistance == -1.0) closeDistance = calculateDistance.apply(MROs.value(j)._2, testPartition(i)._2, distanceType)
        val distance = calculateDistance.mahalanobis(MROs.value(j)._2, testPartition(i)._2, inverse)
        if (distance < closeDistance) closeDistance = distance
      }
      val typical = math.exp(-0.5 * ((closeDistance / umbral) * (closeDistance / umbral)))
      testTipically.append((testPartition(i)._1, typical))
    }
    testTipically.iterator
  }

  def setup(numSamplesTest: Int): Unit ={
    var weightTrain = 0.0
    var weightTest = 0.0
    val numSamplesTrain = train.count
    val numFeatures = train.take(1).map(_.features).length
    val numPartitionMap = train.getNumPartitions

    var logger = Logger.getLogger(this.getClass())
    weightTrain = (8 * numSamplesTrain * numFeatures) / (numPartitionMap * 1024.0 * 1024.0)
    weightTest = (8 * numSamplesTest * numFeatures) / (1024.0 * 1024.0)
    if (weightTrain + weightTest < 1024.0) { //It can be run with one iteration
      numIter = 1
    } else {
      if (weightTrain >= 1024.0) {
        logger.error("=> Train wight bigger than lim-task. Abort")
        System.exit(1)
      }
      numIter = (1 + (weightTest / ((1024.0) - weightTrain)).toInt)
    }


    logger.info("=> NumberIterations \"" + numIter + "\"")

    inc = (numSamplesTest / numIter).toInt
    subdel = 0
    topdel = inc
    if (numIter == 1) { //If only one partition
      topdel = numSamplesTest.toInt + 1
    }
  }

  def runClassify(test: RDD[LabeledPoint], sc: SparkContext) = {

    var testWithIndexLabeled = test.zipWithIndex().map { line => (line._2.toInt, line._1) }.cache
    val labels = testWithIndexLabeled.map { case(ind,v) => (ind, v.label) }.cache
    var testWithIndex = test.zipWithIndex().map { line => (line._2.toInt, line._1.features)}.cache
    var MROsBroadcast: Broadcast[Array[(Int,Vector)]] = null
    var testBroadcast: Broadcast[Array[(Int,Vector)]] = null
    var out1: RDD[(Int, Double)] = null

    var partitionsMROs = 0

    var sumTrainTime = 0.0
    var sumTestTime = 0.0

    setup(test.count().toInt)

    for(i <- 0 until numberEnsembles){

      val sumInitialTrainTime = System.nanoTime
      var MROs = train.sample(withReplacement = true, porcentEnsembles).zipWithIndex().map { line => (line._2.toInt, line._1.features) }.sample(withReplacement = false, porcentMROs).sortByKey().persist()
      if(MROs.count() < 1000.0){partitionsMROs = 1}else{partitionsMROs = ((MROs.count()/1000.0)+1).toInt}
      MROs = MROs.coalesce(partitionsMROs, true)
      val umbral = MROs.mapPartitions(split => MROsDistances(split)).mean()
      sumTrainTime += System.nanoTime-sumInitialTrainTime

      val sumInitialTestTime = System.nanoTime
      MROsBroadcast = broadcastMROs(MROs.collect, MROs.sparkContext)

      var output: RDD[(Int, Double)] = null
      for (i <- 0 until numIter) {
        if (numIter == 1)
          testBroadcast = broadcastMROs(testWithIndex.collect(), sc)
        else
          testBroadcast = broadcastMROs(testWithIndex.filterByRange(subdel, topdel).collect(), testWithIndex.sparkContext)

        if (output == null) {
          output = testWithIndex.mapPartitions{split => tipicality(umbral, MROsBroadcast, split)}.cache
        } else {
          output = output.union(testWithIndex.mapPartitions{split => tipicality(umbral, MROsBroadcast, split)}).cache
        }
        output.count
        //Update the pairs of delimiters
        subdel = topdel + 1
        topdel = topdel + inc + 1
        testBroadcast.destroy
      }


      if (out1 == null) {
        out1 = output.cache
      } else {
        out1 = out1.join(output).map{ case (ind, (v1, v2)) => (ind, v1 + v2) }.persist()
      }

      out1.count
      sumTestTime += System.nanoTime-sumInitialTestTime
      MROs.unpersist()
    }

    val predicts = out1.map { case (ind, v) => (ind, v / numberEnsembles) }.join(labels).values
    predicts
  }
}

object calculateDistance extends Serializable {

  /**
    * Compute the different distances between two vectors
    * @param x: vector
    * @param y: vector
    * @param distanceType: type of distance to be calculated. 1 -> Euclidean, 2 -> Manhattan, 3 -> Chebyshev
    *
    * @return Distance value.
    */

  def apply(x: Vector, y: Vector, distanceType: String): Double = {

    distanceType match {
      case "Euclidean" => euclidean(x, y)
      case "Manhattan" => manhattan(x, y)
      case "Chebyshev" => chebyshev(x, y)
      case _         => euclidean(x, y)
    }
  }


  /** Computes the Euclidean distance between instance x and instance y.
    * The type of the distance used is determined by the value of distanceType.
    *
    * @param x instance x
    * @param y instance y
    * @return Euclidean distance
    */

  private def euclidean(x: Vector, y: Vector): Double = {
    var sum = 0.0
    val size = x.size
    for (i <- 0 until size) sum += (x(i) - y(i)) * (x(i) - y(i))
    Math.sqrt(sum)
  }

  /** Computes the Manhattan distance between instance x and instance y.
    * The type of the distance used is determined by the value of distanceType.
    *
    * @param x instance x
    * @param y instance y
    * @return Manhattan distance
    */

  private def manhattan(x: Vector, y: Vector): Double = {
    var sum = 0.0
    val size = x.size
    for (i <- 0 until size) sum += Math.abs(x(i) - y(i))
    sum
  }

  /** Computes the Chebyshev distance between instance x and instance y.
    * The type of the distance used is determined by the value of distanceType.
    *
    * @param x instance x
    * @param y instance y
    * @return Chebyshev distance
    */

  private def chebyshev(x: Vector, y: Vector): Double = {
    val size = x.size
    val distance = Array.ofDim[Double](size)
    for(i <- 0 until size){
      distance(i) = Math.abs(x(i) - y(i))
    }
    scala.util.Sorting.quickSort(distance)
    distance.last
  }

  def mahalanobis(x: Vector, y: Vector, inverse: Matrix): Double ={
    val size = x.size
    val distance = Array.ofDim[Double](size)
    for(i <- 0 until size){
      distance(i) = x(i)-y(i)
    }
    val vec = Vectors.dense(distance)
    val dist = inverse.multiply(vec)
    /*  for(i<- 0 until size){
        print(dist(i))
      }*/
    var sum = 0.0
    var cont = size-1
    for(i <- 0 until size){
      sum+= dist(cont)*dist(i)
      cont-=1
    }
    //println(sum)
    Math.sqrt(Math.abs(sum))
  }
}
