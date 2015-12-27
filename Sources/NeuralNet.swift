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

// Constants - these can be given to your networks as an instance of
// the neuralnetconfig structure.

// An acceptable amount of error per each category.
private let ERRORPERCATEGORY = 0.04

// The learning rate - how fast network weights change.
private let LEARNINGRATE = -0.3

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
 Network types
 */

/**
 A collection of configurable neural network properties, such as what
 learning rate to use and how many hidden layers to have.

 These can be included in the problem given to the NeuralNet initializer,
 or not. If one is not included, the network will use the default values
 at the top of this file.
 */
public struct NeuralNetConfig {

    /**
     Learning rate of the network. How fast the weights change during
     training. Too high, and backpropagation will jump around without
     settling in an error minimum. Too low, and it may asymptotically
     slow down before reaching one.
     */
    let learningRate: Double

    /**
     Amount of hidden layers in the network.
     */
    let hiddenLayers: Int

    /**
     Number of nodes in each hidden layer of the network.
     If hiddenLayers is zero this can be anything.
     */
    let hiddenNodes: Int

    /**
     An acceptable amount of error per category. The network will be done
     training once the average error in each category is below this number.

     In other words, acceptable error is this times the number of categories,
     and the network is trained once the error across the test set is below
     that number.
     */
    let errorPerCategory: Double
}

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

    /**
     Reset the neuron to its empty initialized state.
     */
    private func reset() {
        activation = 0
        error = 0
        delta = 0
    }
    
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
    let learningRate: Double
    let errorPerCategory: Double
    private var weights = [Matrix<Double>]()
    private var neurons = [[Neuron]]()
    
    var error: Double
    
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
    public init(problem: Classifiable<Category>,
                config: NeuralNetConfig? = nil) {

        srand(UInt32(time(nil)))

        inputs = problem.inputs
        outputs = problem.outputs

        if let values = config {
            hiddenLayers = values.hiddenLayers
            hiddenNodes = values.hiddenNodes
            learningRate = values.learningRate
            errorPerCategory = values.errorPerCategory
        } else {
            hiddenLayers = HIDDENLAYERS
            hiddenNodes = HIDDENNODES
            learningRate = LEARNINGRATE
            errorPerCategory = ERRORPERCATEGORY
        }
        
        // The max error, to ensure backprop always improves after the first
        // pass, is equal to the number of outpus times two.
        error = 2.0 * Double(outputs)

        var lastLayerSize = inputs

        for i in 0...hiddenLayers {
            // Create a 2d array of weights, should have lastLayerSize+1 rows
            // and nextLayerSize collumns
            var newWeights = [[Double]]()
            let nextLayerSize = (i == hiddenLayers ? outputs : hiddenNodes)
            for _ in 0...lastLayerSize { // Range is +1 for bias
                var row = [Double]()
                for _ in 1...nextLayerSize {
                    row.append(myRandom() - 0.5)
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
        
        for (i, layer) in neurons.enumerate() {

            lastActivations.addColumn(1.0)

            let nextActivations = try! matrixProduct(lastActivations,
                                                   b: weights[i])


            // Feed these activation values into the neuron layer
            var activations = [Double]()
            for (j, neuron) in layer.enumerate() {
                neuron.activation = nextActivations[0, j]
                neuron.activate() // sigmoid
                activations.append(neuron.activation)
            }

            lastActivations = Matrix<Double>(array: [activations])!
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
     Internal show method, for debugging purposes.

     Prints all of the weight matrices of the network.
     */
    func show() {
        print("current weights:")
        for weightMatrix in weights {
            print(weightMatrix.contents)
        }
    }

    /**
     Runs a training instance through the network and adjusts weights.

     Returns the error - the delta between expected and actual output.
     */
    func adjustWeightsOnInstance(
                 instance: TrainingInstance<Category>) throws -> Double {

        var error: Double = 0.0

        try compute(instance.input)

        // Adjust weights in output layer
        for i in 0..<outputs {
            let neuron = neurons[hiddenLayers][i]
            let target = (i == instance.category.rawValue ? 1.0 : 0.0)
            let output = neuron.activation

            neuron.delta = output - target

            error += (0.5 * neuron.delta * neuron.delta)

            for j in 0..<weights[hiddenLayers].nRows {
                let preactivation: Double
                if j == weights[hiddenLayers].nRows-1 {
                    preactivation = 1.0
                } else {
                    preactivation = (
                        hiddenLayers == 0 ? instance.input[j] : neurons[
                        hiddenLayers-1][j].activation
                    )
                }
                weights[hiddenLayers][j, i] += (learningRate * preactivation *
                                                neuron.delta * (1 - output) *
                                                output)
                // A lot of this product is repeated. Store?
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

                let ins = (i == 0 ? inputs : neurons[i-1].count)

                for k in 0..<ins {
                    let preactivation = (
                        i == 0 ? instance.input[k] : neurons[i-1][k].activation
                    )
                    weights[i][k, j] += (learningRate * neuron.delta *
                                         (1 - neuron.activation) *
                                         neuron.activation * preactivation)
                }
                
                // Adjust bias weight
                weights[i][ins, j] += (learningRate * neuron.delta *
                                       (1 - neuron.activation) *
                                       neuron.activation * 1.0)
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
        for i in 0..<outputs {
            let instances = training.allOfCategory(Category(rawValue: i)!)
            for instance in instances {
                error += try adjustWeightsOnInstance(instance)
            }
        }
        return error
    }

    /**
     Performs the backpropagation algorithm until the error is acceptable.
     */
    public func backprop(training: TrainingSet<Category>) throws {
        let acceptableError = errorPerCategory * Double(outputs)
        while error > acceptableError {
            error = try trainingEpoch(training)
            //print("error is \(error)")
        }
    }

}
