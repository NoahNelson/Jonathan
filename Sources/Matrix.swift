/**
 This matrix class was created by Ian Kuehne. It can be found in our project at
 https://github.com/ikuehne/Papaya - in AdjacencyMatrix.swift.
 */

/**
 Two-dimensional matrices laid out contiguously in memory.

 Regular two-dimensional Swift arrays consist of an outer array of references to
 inner arrays, which has poor cache performance and thus inefficient access.
 The contiguous arrays implemented in this class have a more efficient memory
 layout at the expense of forcing all rows to be of the same length.
 */
final class Matrix<T> {
    /**
     An array containing the contents of the matrix.  Laid out such that row 0,
     column 0 is adjacent to row 0, column 1 in memory, while row 0, column 0 is
     `self.nCols` spaces away from row 1, column 0.
     */
    var contents: [T]!
    /** The number of rows in the array. */
    var nRows: Int
    /** The number of columns in the array. */
    var nCols: Int!
    
    /**
     Computes the index into `self.contents` corresponding to the given row and
     column.
     
     `self.contents` is laid out such that row 0, column 0 is adjacent to row 0,
     column 1 in memory, while row 0, column 0 is `self.nCols` spaces away from
     row 1, column 0.
     
     - parameter row: The row to index into.
     - parameter col: The column to index into.
     
     - returns: The corresponding index into `self.contents`.
     */
    func index(row row: Int, col: Int) -> Int {
        return col % self.nCols &+ row &* self.nCols
    }
    
    /**
     Creates a new Matrix with a given number of rows and columns, filled
     with a single value.
     
     - parameter rows: The number of rows in the new matrix.
     - parameter cols: The number of columns in the new matrix.
     - parameter repeatedValue: The value with which to fill the new matrix.
     */
    init(rows nRows: Int, cols nCols: Int, repeatedValue: T) {
        self.nRows = nRows
        self.nCols = nCols
        self.contents = [T](count: nRows * nCols, repeatedValue: repeatedValue)
    }
    
    /**
     Creates a new Matrix from the given array of arrays.  Fails if array size
     is uneven.
     
     - parameter array: An array listing the rows of the new matrix; that is,
       `array[row][col]` gives the `row`th row, `col`th column of the new array.
    */
    init?(array: [[T]]) {
        self.nRows = array.count

        for row in array {
            if row.count != array[0].count {
                return nil
            }
        }
        
        if self.nRows >= 1 {
            self.nCols = array[0].count
        } else {
            self.nCols = 0
        }
        
        var optContents = [T?](count: self.nRows * self.nCols,
            repeatedValue: nil)
        for (i, row) in array.enumerate() {
            for (j, item) in row.enumerate() {
                optContents[j % self.nCols &+ i &* self.nCols] = item
            }
        }
        
        self.contents = optContents.map({$0!})
    }

    /**
     Adds a new row to the matrix.

     The row will be added at the end of the matrix, filled with the given
     value.

     - parameter repeatedValue: The value with which to fill the new row.

     - complexity: O(`nCols` * `nRows`)
    */
    func addRow(repeatedValue: T) {
        nRows += 1
        let newRow: [T] = Array<T>(count: nCols, repeatedValue: repeatedValue)
        contents.appendContentsOf(newRow)
    }

    /**
     Adds a new column to the matrix.

     The column will be added at the "right" of the matrix, filled with the
     given value.  Note that this is a very slow operation due to the memory
     layout of the matrices.

     - parameter repeatedValue: The value with which to fill the new row.

     - complexity: O(`nCols` * `nRows`)
    */
    func addColumn(repeatedValue: T) {
        nCols = nCols + 1
        var newContents: [T] = Array<T>(count: nRows * (nCols + 1),
                                        repeatedValue: repeatedValue)

        // Put the old items in their new positions.
        for (index, item) in contents.enumerate() {
            newContents[index + index / (nCols - 1)] = item
        }

        // Fill in the new items.
        for row in 0..<nRows {
            newContents[index(row: row, col: nCols - 1)] = repeatedValue
        }

        contents = newContents
    }

    /**
     Removes a given row from the matrix.

     - parameter row: The row to be removed.

     - returns: `Bit.One` if the row was in the matrix, or `.Zero` otherwise.

     - complexity: O(`nRows` * `nCols`)
    */
    func removeRow(row: Int) -> Bit {
        if row >= nRows {
            return Bit.Zero
        }
        var newContents = [T]()
        let beforeRow = contents[0..<(row * nCols)]
        let afterRow = contents[((row + 1) * nCols)..<(nRows * nCols)]
        newContents.appendContentsOf(beforeRow)
        newContents.appendContentsOf(afterRow)
        contents = newContents
        nRows = nRows - 1
        return Bit.One
    }

    /**
     Removes a given column from the matrix.  Returns `Bit.One` if the
     column was in the matrix, or `.Zero` otherwise.

     - parameter col: The column to be removed.

     - returns: `Bit.One` if the column was in the matrix, or `.Zero` otherwise.

     - complexity: O(`nRows` * `nCols`)
    */
    func removeColumn(col: Int) -> Bit {
        if col >= nCols {
            return Bit.Zero
        }
        var newContents = [T]()
        for (index, item) in contents.enumerate() {
            if (index % nCols) != 0 {
                newContents.append(item)
            }
        }
        contents = newContents
        nCols = nCols - 1
        return Bit.One
    }
    
    /**
     Retrieves or sets the element at a given row and column of the matrix.
     
     - parameter row: The row to use.
     - parameter col: The column to use.
     
     - returns: If `get`, then the element at the given row and column.  If
       `set`, then nothing.
     */
    subscript(row: Int, col: Int) -> T {
        get {
            return self.contents[self.index(row: row, col: col)]
        }
        set(newValue) {
            self.contents[self.index(row: row, col: col)] = newValue
        }
    }
}
