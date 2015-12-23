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

}

/**
 A structure which describes what your classification problem is.

 A classification problem must specify three things - input dimension,
 number of output categories, and a list of valid output categories.
 */
public struct Classifiable {

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

    /**
     Includes the set of categories, which is anything that conforms to the
     Categorical protocol.
     */
    let categories: Categorical

}

/**
 A training case for a classification problem.

 This is a sample instance of your classification problem. Jonathan will use
 many of these to train.

 It consists of an input array of doubles, and the output that that array
 should map to.
 */
public struct TrainingCase<T: Categorical> {
    let input: [Double]

    let category: T
}
    
public class TestSet {
    
    //  A class for classification problem training data.
    //  Has a mapping of example inputs to their category.
    
    var categories: [Int: [[Double]]]
    // Should this be private? And just define an iterator or something for
    // training code to use? Leave internal for now.
    
    public init() { categories = [:] }
    
    public func addPathToCategory(path: [Double], category: Int) {

        // Either the category number already exists in the testset, or not.
        if let _ = categories[category] {
            categories[category]!.append(path) // Better way to write this?
        }
        else {
            categories[category] = [path]
        }
        
    }
    
    // TODO: ReadFromFile function
    
    public func show() {
        for (category, items) in categories {
            print("\(category) : \(items)") ;
        }
    }
    
} 
