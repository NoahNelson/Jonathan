/**
 TestSet.swift - provides protocols that define precisely what a classification
 problem is, as well as TrainingSets for them.
 */


/**
 protocol Categorical: This defines what it means to be a set of categories
 for a classification problem. Any Categorical type can be used as an output
 space for a classification problem.

 To be categorical, you only need a way of constructing an instance of a
 category using a raw integer value.

 Thus, enumerations work especially well as Categorical types.
 */
public protocol Categorical {
 
    /**
     We must be able to initialize a category using a valid raw integer value.

     - parameter rawValue: The integer number of which case to create.

     Note that these should start at 0 and count up by 1. So if there are n
     cases in the Categorical, this ought to create a valid case for any
     rawValue parameter in 0..<n
     */
    init?(rawValue: Int)

    /**
     We must be able to get the raw value of a category in the numbering.
     */
    var rawValue: Int { get }

}

/**
 A structure which describes what your classification problem is.

 A classification problem must specify three things - input dimension,
 number of output categories, and a list of valid output categories.
 */
public struct Classifiable<Category: Categorical> {

    /**
     Includes an input dimension - this is the number of doubles in an input
     array for your classification problem.
     */
    let inputs: Int

    /**
     Includes the size of the output space. This is the number of possible
     categories you could put an input into.
     */
    let outputs: Int
}

/**
 A training case for a classification problem.

 This is a sample instance of your classification problem. Jonathan will use
 many of these to train.

 It consists of an input array of doubles, and the output that that array
 should map to.
 */
public struct TrainingInstance<Category: Categorical> {

    let input: [Double]

    let category: Category
}

/**
 A structure representing a set of training data.

 Includes a read-only array of training cases, and some methods to add cases,
 as well as the input and output numbers.
 */
public struct TrainingSet<T: Categorical> {

    let inputs: Int

    let outputs: Int
    
    /**
     A public get private set array of training instances.

     Currently, you can add instances but you can't remove them.
     */
    public private(set) var instances = [TrainingInstance<T>]()

    /**
     Adds a training instance to the training set.

     - parameter instance: The TrainingInstance to add to the dataset.

     - throws: `NeuralNetError.IncorrectInputSize` if the training instance has
       the incorrect size of input array.
     */
    public mutating func addInstance(instance: TrainingInstance<T>) throws {
        guard inputs == instance.input.count else {
            throw NeuralNetError.IncorrectInputSize
        }

        instances.append(instance)
    }

    // TODO: ReadFromFile function
    
    public func show() {
        for instance in instances {
            print("\(instance.input) : \(instance.category)")
        }
    }
    
} 
