# Bagging-RandomMiner

BaggingRandomMiner [1] is a model based on the methodology of ensembles for the classification of a single class, which bases its training on randomization. This algorithm is a distributed version proposed in [2], which managed to show very good performance in the PRIDE data set (Personal RIsk DEtection) where the best performance in the state of the art had it OCKRA (One-Class K-means with Randomly -projected features) proposed in [3]

## Example (MLlib)

```import org.apache.spark.mllib.classification._

ensemblePercent = 0.1 //(0 - 1)
MROsPercent = 0.10 //(0 - 1)
nEnsembles = 10 //(Number of models)
typeDistance = 3 //(1 -> euclidean, 2 -> Manhattan, 3 -> Chebyshev)

// Data must be cached in order to improve the performance

val RandomMinerModel = new RandomMiner(trainingData, // RDD[LabeledPoint]
                            nEnsembles, 
                            ensemblePercent, 
                            MROsPercent, 
                            typeDistance)

val predicted =  RandomMinerModel.runClassify(testingData, sc)
```

## Reference

```
[1] Camiña, J.B., Medina-Pérez, M.A., Monroy, R., Loyola-González, O., PereyraVillanueva, L.A., González-Gurrola, L.C.: Bagging-randomminer: A one-class classifier for file access-based masquerade detection. Machine Vision and Applications (2018)
[2] Luis Pereyra, Diego García-Gil, Francisco Herrera, Luis C. González-Gurrola, Jacinto Carrasco, Miguel Angel Medina-Pérez  and Raúl Monroy, CAEPI (2018)
[3] Rodríguez, J., Barrera-Animas, A.Y., Trejo, L.A., Medina-Pérez, M.A., Monroy, R. Ensemble of one-class classifiers for personal risk detection based on wearable sensor data. Sensors 16(10) (2016)
```
