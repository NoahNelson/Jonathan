enum BitCases: Int, Categorical {
    case Zero, One
}

let xor = Classifiable<BitCases>(inputs: 2, outputs: 2)

let cases = [TrainingInstance<BitCases>(input: [0.0, 0.0], category: .Zero),
             TrainingInstance<BitCases>(input: [0.0, 1.0], category: .One),
             TrainingInstance<BitCases>(input: [1.0, 0.0], category: .One),
             TrainingInstance<BitCases>(input: [1.0, 1.0], category: .Zero)]

var xorTraining = TrainingSet<BitCases>(inputs: 2, outputs: 2, instances: cases)

xorTraining.show()

let inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

let network = NeuralNet<BitCases>(problem: xor)

for input in inputs {
    print(try! network.compute(input))
}
