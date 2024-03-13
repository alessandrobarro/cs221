#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    words = x.split()
    FeatureVector = {}
    for word in words:
        if word in FeatureVector:
            FeatureVector[word] += 1
        else:
            FeatureVector[word] = 1
    return FeatureVector
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    def dotProduct(v: FeatureVector, w: WeightVector) -> float:
        return sum(v[feature] * w.get(feature, 0) for feature in v)

    def updateWeights(weights: WeightVector, features: FeatureVector, y: int, eta: float):
        for feature, value in features.items():
            if (dotProduct(features, weights) * y) <= 1:
                weights[feature] = weights.get(feature, 0) + eta * y * value
        return weights

    for _ in range(numEpochs):
        for x, y in trainExamples:
            features = featureExtractor(x)
            weights = updateWeights(weights, features, y, eta)
    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        phi = {}
        for item in random.sample(list(weights), random.randint(1, len(weights))):
            phi[item] = random.randint(1, 100)
        if dotProduct(weights, phi) > 1:
            y = 1
        else:
            y = 0
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        x = x.replace(' ', '')
        feature_vector = {}
        for i in range(len(x) - n + 1):
            ngram = x[i:i + n]
            if ngram in feature_vector:
                feature_vector[ngram] += 1
            else:
                feature_vector[ngram] = 1
        return feature_vector
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3e:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))

############################################################
# Problem 5: k-means
############################################################




def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    Perform K-means clustering on |examples|, where each example is a sparse feature vector.

    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)
    import collections

    def initializeCentroids(samples, num_clusters):
        return [sample.copy() for sample in random.sample(samples, num_clusters)]

    def assignToClusters(samples, num_clusters, centroids):
        assignments = [random.randint(0, num_clusters-1) for _ in samples]
        return assignments

    def calculateSquared(samples):
        squared_samples = []
        for sample in samples:
            squared_sample = collections.defaultdict(float)
            for key, value in sample.items():
                squared_sample[key] = value * value
            squared_samples.append(squared_sample)
        return squared_samples

    centroids = initializeCentroids(examples, K)
    cluster_assignments = assignToClusters(examples, K, centroids)
    examples_squared = calculateSquared(examples)

    previous_assignments = None

    for epoch in range(maxEpochs):
        centroids_squared = calculateSquared(centroids)
        distances = [0 for _ in examples]

        for i, example in enumerate(examples):
            min_distance = float('inf')
            for j, centroid in enumerate(centroids):
                distance = sum(examples_squared[i].values()) + sum(centroids_squared[j].values())
                for key in (example.keys() & centroid.keys()):
                    distance += -2 * example[key] * centroid[key]
                if distance < min_distance:
                    min_distance = distance
                    cluster_assignments[i] = j
                    distances[i] = min_distance

        if previous_assignments == cluster_assignments:
            break

        centroids, cluster_counts = updateCentroids(cluster_assignments, examples, centroids)

        previous_assignments = cluster_assignments[:]

    return centroids, cluster_assignments, sum(distances)

def updateCentroids(assignments, samples, centroids):
    cluster_counts = [0 for _ in centroids]
    for i, _ in enumerate(centroids):
        for key in centroids[i]:
            centroids[i][key] = 0.0
    for i, sample in enumerate(samples):
        cluster_counts[assignments[i]] += 1
        for key, value in sample.items():
            centroids[assignments[i]][key] += value
    for i, centroid in enumerate(centroids):
        for key in centroid:
            centroid[key] /= cluster_counts[i]
    return centroids, cluster_counts
    # END_YOUR_CODE
