/**
 Network.swift - this file defines a neural network class and related errors.

 Neural networks can compute a function from an array of doubles and train
 themselves using the backpropagation algorithm on a test set.
 */

#if os(Linux)
import Glibc
#else
import Darwin.C
#endif

// Constants - eventually these will be read in from an rc file

// The starting value of a network's error.
private let MAXERROR: Double = 1000

// An acceptable amount of error per each category.
private let ERRORPERCATEGORY = 0.04

// The learning rate - how fast network weights change.
private let LEARNINGRATE = -0.5

// The number of hidden layers in each network
private let HIDDENLAYERS = 1

// The number of nodes in each hidden layer
private let HIDDENNODES = 2

/**
 Errors for neural networks
 */
public enum NeuralNetError: ErrorType {
    case IncorrectInputSize
}

/**
 Helper and math functions
 */

/**
 random number function
 question: does this need to be seeded?

 - returns: random double between 0 and 1
 */
private func myRandom() -> Double {
#if os(Linux)
    return Double(random()) / Double(UINT32_MAX)
#else
    return Double(arc4random()) / Double(UINT32_MAX)
#endif
}

/**
 Sigmoid activation function for artificial neurons.

 - parameter x: a double representing unprocessed activation.

 - returns: the result of the sigmoid activation function on x.
 */
private func sigmoid(x: Double) -> Double {
    let exp = pow(Double(M_E), -x)
    return (1.0/(1.0 + exp))
}


/**
 Network classes
 */


/**
 An artificial Neuron, to be arrayed in a network.

 Stores an activation value as well as error and deltas used in backprop.

 I changed this to a class to get reference behavior, which allows for easy
 mutations and stuff.
 */
private class Neuron {

    /**
     The current activation of the neuron
     */
    private var activation: Double = 0

    /**
     Run the activation through the sigmoid function.
     */
    private func activate() { activation = sigmoid(activation) }
    
    /**
     Error between the desired and actual output of the neuron
     */
    private var error: Double = 0

    /**
     Variable representing deltaTotalError/deltaNeuronActivation,
     used in backprop algorithm
     */
    private var delta: Double = 0
    
}

/**
 An artificial neural network object for classification problems.

 Computes a function from an array of doubles to a set of categories,
 which may have different sizes.

 The input represents an instance of a classification problem, and the output
 represents highest probability of that instance being a given class.

 Has code to train itself using backpropagation, given a test set.
 */
public class NeuralNet<Category: Categorical> {
    
    // TODO: figure out which of these need to be public.
    let inputs: Int
    let outputs: Int
    let hiddenLayers: Int
    let hiddenNodes: Int
    private var weights = [Matrix<Double>]()
    private var neurons = [[Neuron]]()
    
    var error: Double
    let ErrPerCategory = ERRORPERCATEGORY
    
    /**
     Creates a new, untrained neural network to solve a given classification
     problem. Number of inputs and outputs are given by that classification
     problem. Number of hidden layers and nodes per hidden layer are constants
     that are currently defined at the top of this file but will eventually
     be read from a .jonrc file.

     Untrained means random weights - likely to not be suited to solving the
     classification problem until it's been fed a training set.

     - parameter problem: A classifiable structure that describes the problem
       this neural network ought to solve.
     */
    public init(problem: Classifiable<Category>) {

        srand(UInt32(time(nil)))

        inputs = problem.inputs
        outputs = problem.outputs
        hiddenLayers = HIDDENLAYERS
        hiddenNodes = HIDDENNODES
        
        // The max error, to ensure backprop always improves after the first
        // pass, is equal to the number of outpus times two.
        error = 2.0 * Double(outputs)

        var lastLayerSize = inputs

        for i in 0...hiddenLayers {
            // Create a 2d array of weights, should have lastLayerSize rows
            // and nextLayerSize collumns
            var newWeights = [[Double]]()
            var nextLayerSize = 0
            for _ in 1...lastLayerSize {
                var row = [Double]()
                nextLayerSize = (i == hiddenLayers ? outputs : hiddenNodes)
                for _ in 1...nextLayerSize {
                    row.append(myRandom())
                }
                newWeights.append(row)
            }
            let matrix = Matrix<Double>(array: newWeights)!
            weights.append(matrix)
            
            // And, add an array of neurons for this layer.
            var nextLayer = [Neuron]()
            for _ in 1...nextLayerSize {
                nextLayer.append(Neuron())
            }
            neurons.append(nextLayer)
            lastLayerSize = nextLayerSize
        }
    }
    /*
    /**
     This will be useful for reading in networks from files, etc.
     */
    init(withWeights weights: [[[Double]]]) {
        self.neurons = []
        inputs = weights[0][0].count - 1
        hiddenLayers = weights.count - 1
        nodesPerHiddenLayer = (hiddenLayers == 0 ? 0 :
                               weights[hiddenLayers][0].count - 1) // Bias!
        
        for layer in weights {
            var newLayer = [Neuron]()
            for neuron in layer {
                newLayer.append(Neuron(withWeights: neuron))
            }
            self.neurons.append(newLayer)
        }
    }
    */
    
    /**
     Computes the most likely category of a given input array of Doubles.

     - parameter input: An array of doubles representing an instance of the
       classification problem.

     - returns: The most likely category of the input.

     - throws: `NeuralNetError.IncorrectInputSize`
     */
    public func compute(input: [Double]) throws -> Category {

        guard input.count == inputs else {
            throw NeuralNetError.IncorrectInputSize
        }

        var lastActivations = Matrix<Double>(array: [input])!
        for i in 0..<neurons.count { // Not enumerate so we can mutate

            let nextActivations = try! matrixProduct(lastActivations,
                                                   b: weights[i])

            // Feed these activation values into the neuron layer
            var layer = neurons[i]
            for j in 0..<layer.count {
                layer[j].activation = nextActivations[0, j]
                layer[j].activate()
            }
            lastActivations = nextActivations
        }
        
        //  Now we can return the contents of the last layer of neurons -
        //  These are the output neurons.
        
        var (maxIndex, maxOutput) = (-1, -1.0)

        for (i, output) in neurons[hiddenLayers].enumerate() {
            if output.activation > maxOutput {
                maxOutput = output.activation
                maxIndex = i
            }
        }

        return Category(rawValue: maxIndex)!
    }

    /**
     Runs a training instance through the network and adjusts weights.

     Returns the error - the delta between expected and actual output.
     */
    private func adjustWeightsOnInstance(
                 instance: TrainingInstance<Category>) throws -> Double {

        //print("current weights:")
        for weightMatrix in weights {
         //   print(weightMatrix.contents)
        }

        var error: Double = 0.0

        try compute(instance.input)

        // Adjust weights in output layer
        for i in 0..<outputs {
            let neuron = neurons[hiddenLayers][i]
            let target: Double = (i == instance.category.rawValue ? 1 : 0)
            let output = neuron.activation

            neuron.delta = output - target

            error += (0.5 * neuron.delta * neuron.delta)

            for j in 0..<weights[hiddenLayers].nRows {
                let preactivation = (
                    hiddenLayers == 0 ? instance.input[j] : neurons[
                    hiddenLayers-1][j].activation
                )
                weights[hiddenLayers][i, j] += (LEARNINGRATE * preactivation *
                                                neuron.delta * (1 - output) *
                                                output)
            }
        }

        // Adjust weights of the hidden layers

        for i in (0..<hiddenLayers).reverse() {
            for j in 0..<neurons[i].count {
                let neuron = neurons[i][j]
                neuron.delta = 0
                
                for k in 0..<neurons[i+1].count {
                    let postpartner = neurons[i+1][k]
                    neuron.delta += (postpartner.delta * postpartner.activation
                                     * (1 - postpartner.activation) *
                                     weights[i+1][j, k])
                }

                for k in 0..<(i == 0 ? inputs : neurons[i-1].count) {
                    let preactivation = (
                        i == 0 ? instance.input[k] : neurons[i-1][k].activation
                    )
                    weights[i][i, k] += (LEARNINGRATE * neuron.delta *
                                         (1 - neuron.activation) *
                                         neuron.activation * preactivation)
                }
            }
        }

        return error
    }

    /**
     Performs one epoch of the backpropagation algorithm on a given training
     set, returning the total error.
     */
    private func trainingEpoch(
            training: TrainingSet<Category>) throws -> Double {
        var error = 0.0
        for instance in training.instances {
            error += try adjustWeightsOnInstance(instance)
        }
        return error
    }

    /**
     Performs the backpropagation algorithm until the error is acceptable.
     */
    public func backprop(training: TrainingSet<Category>) throws {
        let acceptableError = ERRORPERCATEGORY * Double(outputs)
        while error > acceptableError {
            error = try trainingEpoch(training)
            print("error is \(error)")
        }
    }

    
    /*
    private func trainingEpoch(testset: TestSet) -> Double {
        
        //  Performs one step of the backpropagation algorithm
        //  on a given test set.
        //  Returns the error of the network.
        
        
        var totalError: Double = 0
        
        for (category, inputs) in testset.categories {
            for input in inputs {
                                
                compute(input)
                
                //  Adjust weights of the output layer
                for (i, neuron) in neurons[hiddenLayers].enumerate() {
                    let target = (category == i ? 1.0 : 0.0)
                    let output = neuron.activation
                    neuron.error = 0.5 * (target - output) * (target - output)
                    // Does this need to be stored in neuron?
                    
                    totalError += neuron.error
                    
                    neuron.delta = output - target
                    
                    for (j, prepartner) in neurons[hiddenLayers-1].enumerate() {
                        neuron.incrementWeight(j, inc: (LEARNINGRATE *
                                prepartner.activation * (output - target) *
                                (1 - output) * output))
                    }
                    
                }
                
                //  Adjust weights of the hidden layers
                
                //  TODO: This for loop style will be removed from language
                for var i = hiddenLayers-1; i >= 0; --i {
                    for (j, neuron) in neurons[i].enumerate() {
                        let outputH = neuron.activation
                        
                        var sum = 0.0
                        
                        //  Calculate sum
                        
                        for (_, postpartner) in neurons[i+1].enumerate() {
                            let outputI = postpartner.activation
                            sum += (postpartner.delta * postpartner.weights[j]
                                    * outputI * (1 - outputI))
                        }
                        
                        neuron.delta = sum
                        
                        
                        if (i == 0) {
                            for (k, input) in input.enumerate() {
                                
                                neuron.incrementWeight(k, inc: (LEARNINGRATE *
                                        sum * (1 - outputH) * outputH * input))
                            }
                        }
                        else {
                            for (k, prepartner) in neurons[i-1].enumerate() {
                                
                                neuron.incrementWeight(k, inc: (LEARNINGRATE *
                                        sum * (1 - outputH) * outputH *
                                        prepartner.activation))
                            }
                        }
                        
                    }
                }
                
            }
        }
        
        self.error = totalError
        return error ;
        
    }
    
    public func backprop(testset: TestSet) {
        
        //  Performs backpropagation algorithm on the network, given a test set.
        
        var currentError = 1000.0 ;
        var lastError: Double = 1001.0 ;
        var staleness = 0 ;
        var gens = 0 ;
        let ErrorThreshold = (ErrorThresholdPerCategory *
                              Double(testset.categories.count))
        
        while (currentError > ErrorThreshold) {
            
            currentError = trainingEpoch(testset) ;
            
            if currentError >= lastError {
                staleness += 1 ;
            }
            else {
                staleness = 0 ;
            }
            
            lastError = currentError ;
            
            gens += 1
        }
    }
    */
}
