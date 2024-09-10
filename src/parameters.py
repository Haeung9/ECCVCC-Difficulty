
class codeParameters:
    def __init__(self, blockLength = 32, rowDegree = 4, colDegree = 3, hammingWeigthLow = 0, hammingWeigtHigh = 32, decisionStep = 2):
        self.block_length = blockLength; self.row_deg = rowDegree; self.col_deg = colDegree
        self.redundancy = int(self.block_length * self.col_deg / self.row_deg)
        if not (blockLength * colDegree == self.redundancy * rowDegree):
            raise ValueError("Invalid code parameters")
        if (hammingWeigtHigh > self.block_length) or (hammingWeigthLow < 0):
            raise ValueError("Invalid Hamming Weigth region")
        self.hammingWeigthLow = hammingWeigthLow
        self.hammingWeigthHigh = hammingWeigtHigh
        self.decisionStep = decisionStep