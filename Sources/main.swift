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

/*let matrix1 = Matrix<Int>(array: [[1, 2], [2, 1]])!
let matrix2 = Matrix<Int>(array: [[3, -1], [0, 2]])!

let matrix3 = try! matrixProduct(matrix1, b: matrix2)
print(matrix3.contents)

let matrix4 = Matrix<Double>(array: [[1.2, 2.3], [-0.5, 0.3], [0, 1.0]])!
let matrix5 = Matrix<Double>(array: [[-1.3, 0, 0.4], [-2.0, 1.0, 0]])!
let matrix6 = try! matrixProduct(matrix4, b: matrix5)
print(matrix6.contents)
// brief tests of matrix multiplication, works
*/
