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
private let LEARNINGRATE = -0.4

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
 */
private struct Neuron {

    /**
     The current activation of the neuron
     */
    private var activation: Double = 0

    /**
     Run the activation through the sigmoid function.
     */
    private mutating func activate() { activation = sigmoid(activation) }
    
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
public class NeuralNet {
    
    // TODO: figure out which of these need to be public.
    let inputs: Int
    let outputs: Int
    let hiddenLayers: Int
    let hiddenNodes: Int
    private var weights: [Matrix<Double>]
    private var neurons: [[Neuron]]

    var error: Double
    let ErrPerCategory = ERRORPERCATEGORY
    
    public init(problem: Classifiable, hiddenNodes: Int, layers: Int = 0) {

        inputs = problem.inputs
        outputs = problem.outputs
        hiddenLayers = layers
        self.hiddenNodes = hiddenNodes
        
        // The max error, to ensure backprop always improves the first pass,
        // is equal to the number of outpus times two.
        error = 2.0 * Double(outputs)

        if layers == 0 {
            // Simple perceptron case, we only need weights from in to output.
            var weights = [[Double]]()
            for _ in 1...inputs {
                var cols = [Double]()
                for _ in 1...outputs {
                    cols.append(myRandom() - 0.5)
                }
                weights.append(cols)
            }
            self.weights = [Matrix<Double>(array: weights)!]
            var outputNeurons = [Neuron]()
            for _ in 1...outputs {
                outputNeurons.append(Neuron())
            }
        }
    }
    
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
    
    
    /* Increases the number of outputs. Maybe not useful in most applications.
    func addNewOutput() {
        
        //  Sticks a new output in, with random weights
        
        if hiddenLayers == 0 {
            self.neurons[hiddenLayers].append(Neuron(numWeights: inputs + 1))
        }
        
        else {
            self.neurons[hiddenLayers].append(
                        Neuron(numWeights: nodesPerHiddenLayer + 1))
        }
        
    }
    */
    
    // Function to compute output from inputs, using current weights.
    public func compute(inputs: [Double]) -> [Double] {
                
        if inputs.count != self.inputs {
            print("different number of inputs than expected!")
            return []
            // TODO: Error-handling behavior
        }
        
        
        for (i, layer) in neurons.enumerate() {
            if (i == 0) {
                //  Run inputs into first hidden layer
                for neuron in layer {
                    neuron.activation = 0
                    
                    for (j, input) in inputs.enumerate() {
                        neuron.activation += neuron.weights[j] * input
                    }
                    neuron.activation += (neuron.weights[neuron.weights.count-1]
                                          * 1.0) //  Bias
                    
                    neuron.activation = sigmoid(neuron.activation)
                }
            }
            
            else {
                
                for neuron in layer {
                    neuron.activation = 0
                    
                    if (neuron.weights.count-1 != neurons[i-1].count) {
                    //print("Different number of weights than inputs in \(i)!")
                        // Should throw an error
                    }
                    
                    for (j, presynaptic) in neurons[i-1].enumerate() {
                        neuron.activation += (neuron.weights[j] *
                                              presynaptic.activation)
                    }
                    
                    neuron.activation += (1.0 *
                                neuron.weights[neuron.weights.count-1])
                                // Add bias
                    neuron.activation = sigmoid(neuron.activation)
                }
            }
            
        }
        
        //  Now we can return the contents of the last layer of neurons -
        //  These are the output neurons.
        
        var outputs = [Double]()
        
        for neuron in neurons[hiddenLayers] {
            outputs.append(neuron.activation)
        }
        
        return outputs
    }
    
    
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
}
