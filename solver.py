import numpy as np
import argparse


# Linear Equations Solver
# Stage 1/5: Real and Simple

"""
a, b = [float(x) for x in input().split()]

if a != 0:
    print(b / a)
"""

# Stage 2/5: X and Y

"""
a, b, c, d, e, f = [float(x) for _ in range(2) for x in input().split()]

y = (f - c * d / a) / (e - b * d / a)
x = (c - b * y) / a

print(x, y)
"""

# Stage 3/5: Equation Time
# https://hyperskill.org/projects/144/stages/778/implement

"""
parser = argparse.ArgumentParser(description="This program solves a system of linear equations.")
parser.add_argument("--infile")
parser.add_argument("--outfile")
args = parser.parse_args()

matrix_ = []

with open(args.infile, 'r') as file:
    for line in file:
        line = line.replace('/n', '').split()
        row_ = [float(x) for x in line]
        matrix_.append(row_)

matrix_ = matrix_[1:]
matrix_array = np.array(matrix_)


class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.n = len(matrix)

    @property
    def solution(self):
        print("\nStart solving the equation.")
        m = self.matrix
        n = self.n

        # iterate over columns, ascendant
        for col in range(n):
            a = m[col, col]
            if a not in [0, 1]:
                print(f"{1 / a} * R{col + 1} -> R{col + 1}")
                m[col, :] = m[col, :] / a
                print(m, '\n')
            if a:
                # iterate over rows, ascendant
                for row in range(col + 1, n):
                    c1 = - m[row, col]
                    print(f"{c1} * R{col + 1} + R{row + 1} -> R{row + 1}")
                    m[row, :] = c1 * m[col, :] + m[row, :]
                    print(m, '\n')
        # iterate over columns, descendant
        for col in range(n - 1, 0, -1):
            # iterate over rows, descendant
            for row in range(col - 1, -1, -1):
                c2 = - m[row, col]
                print(f"{c2} * R{col + 1} + R{row + 1} -> R{row + 1}")
                m[row, :] = c2 * m[col, :] + m[row, :]
                print(m, '\n')
        return [m[x, n] for x in range(n)]


new_matrix = Matrix(matrix_array)
result = new_matrix.solution
print(f"The solution is {tuple(result)}.")

with open(args.outfile, 'w') as file:
    for el in result:
        file.write(str(el) + '\n')

print(f"Saved to out.txt")
"""

# Stage 4/5: Where Things Can Get complicated
# https://hyperskill.org/projects/144/stages/779/implement

# Stage 5/5: Complex Numbers
# https://hyperskill.org/projects/144/stages/780/implement

parser = argparse.ArgumentParser(description="This program solves a system of linear equations.")
parser.add_argument("--infile")
parser.add_argument("--outfile")
args = parser.parse_args()

matrix_ = []

# read matrix from file
with open(args.infile, 'r') as file:
    for line in file:
        line = line.replace(',', '').replace('\'', '').replace('/n', '').split()
        print(f"{line}")
        line_type = "float"
        for number in line:
            if "j" in number:
                line_type = "complex"
        if line_type == "complex":
            row_ = [complex(x) for x in line]
        else:
            row_ = [float(x) for x in line]
        matrix_.append(row_)

rows_number = len(matrix_) - 1
cols_number = len(matrix_[1])
matrix_ = matrix_[1:]

# matrix -> create numpy array
matrix_array = np.array(matrix_)

# * print(f"m = rows_number {rows_number}")  *
# * print(f"n = cols_number {cols_number}")  *
# * print(f"matrix_array\n{matrix_array}")  *


class Matrix:
    def __init__(self, matrix, rows, cols):
        self.matrix = matrix
        self.rows = rows
        self.cols = cols
        self.matrix_significant = matrix
        self.rows_significant = rows
        self.swap_list = []
        self.solution_list = []
        self.solution = self.solver()

    # if there is a zero coefficient on main diagonal -> look for a non-zero coefficient below and to the right ...
    # ... if a non-zero coeff is found : ...
    # ... manipulate matrix_significant -> swap rows and columns if a non-zero coefficient is found
    def analyze_coeff(self, mat, m, n, col):
        last = col
        while last < m:
            # look for a non_zero coefficient below
            for x in range(last + 1, m):
                # swap rows if a non-zero coefficient is found below ...
                # ... and modify matrix_significant
                if mat[x, last]:
                    print(f"R{last + 1} <-> R{x + 1}")
                    mat[[last, x], :] = mat[[x, last], :]
                    self.matrix_significant = mat
                    return True
            # look for a non-zero coefficient to the right
            for y in range(last + 1, n - 1):
                # swap cols if a non-zero coefficient is found to the right...
                # ... add the changed cols to the self.swap list ...
                # ... and return the resulted matrix
                if mat[last, y]:
                    self.swap_list.append((last, y))
                    mat[:, [last, y]] = mat[:, [y, last]]
                    self.matrix_significant = mat
                    return True
            last += last
        # if the current row is an all zero row (coefficients and constant) ...
        # ... look for a non-zero row and swap with it
        for x in range(m - 1):
            if not sum(mat[x]):
                for y in range(x + 1, m):
                    if mat[y, n - 1]:
                        mat[[x, y], :] = mat[[y, x], :]
                        self.matrix_significant = mat
        self.matrix_significant = mat
        self.rows_significant = self.matrix_significant.shape[0]
        return False

    # check if the matrix has solutions (none, one or infinitely)
    def check_significant(self):
        solution = "_"
        significant = True

        mat = self.matrix_significant

        # calculate the number of all zero rows (coefficients and constant == zero)
        zero_rows = sum([np.all(mat[x] == 0) for x in range(self.rows)])

        # calculate the number of significant rows (non-zero rows)
        rows_s = self.rows - zero_rows
        self.rows_significant = rows_s

        # number of columns = number of variables
        # case: number of significant rows > number of columns => No solutions
        if rows_s > self.cols - 1:
            solution = "No solutions"
            significant = False

        # case: number of significant rows < number of columns
        elif rows_s < self.cols - 1:
            solution = "Infinitely many solutions"
            # if on one row the coefficients sum == 0 and the constant != 0 => No solutions
            for x in range(rows_s):
                if sum(mat[x, :self.cols - 2]) == 0 and mat[x, self.cols - 1] != 0:
                    solution = "No solutions"
                    significant = False

        # case: number of significant rows == number of columns
        elif rows_s == self.cols - 1:
            for x in range(rows_s):
                coeff_sum = mat[x, :self.cols - 1].sum()
                constant = mat[x, self.cols - 1]
                if not coeff_sum and constant:
                    if rows_s < self.rows:
                        solution = "Infinitely many solutions"
                        significant = False
                    else:
                        solution = "No solutions"
                        significant = False
                # if on one row a constant is zero => ...
                # ... the matrix has a solution and the correspondent variable is zero
                elif coeff_sum and not constant:
                    significant = True

        return solution, significant

    @staticmethod
    def gauss_elimination_cols_ascendant(mat, col, m):
        # manipulate rows, ascendant -> Row Echelon Form
        for row in range(col + 1, m):
            c1 = - mat[row, col]
            if c1:
                # print the operation on row
                print(f"{c1} * R{col + 1} + R{row + 1} -> R{row + 1}")
                mat[row, :] = c1 * mat[col, :] + mat[row, :]
        return mat

    @staticmethod
    def gauss_elimination_cols_descendant(mat, col):
        # manipulate rows, descendant -> Reduce Row Echelon Form
        for row in range(col - 1, -1, -1):
            c2 = - mat[row, col]
            if c2:
                print(f"{c2} * R{col + 1} + R{row + 1} -> R{row + 1}")
                mat[row, :] = c2 * mat[col, :] + mat[row, :]
        return mat

    # after solving the matrix, swap back columns in order to ...
    # ... keep the order of variables in the final solution
    def swap_cols(self):
        self.swap_list.reverse()
        mat = self.matrix_significant
        for t in self.swap_list:
            mat[:, [t[0], t[1]]] = mat[:, [t[1], t[0]]]
        self.matrix_significant = mat

    # "main method" -> analyze each column
    def solver(self):
        print("\nStart solving the equation.\nRows manipulation:")
        mat = self.matrix
        m = self.rows
        n = self.cols
        significant = True
        a = mat[0, 0]

        # gauss elimination - iterate ascendant over variables columns
        for col in range(n - 1):
            # check the coefficient 'a' on the main diagonal
            if col < m:
                a = mat[col, col]
            # if a == 0, look for a non-zero coefficient with the function check_coeff
            if not a:
                result = self.analyze_coeff(mat, m, n, col)
                mat = self.matrix_significant
                if not result:
                    solution, significant = self.check_significant()
                    return solution
            # else if a != 1, divide the row by a
            if a not in [1]:
                print(f"{1 / a} * R{col + 1} -> R{col + 1}")
                mat[col, :] = mat[col, :] / a
            # if a == 1, manipulate the matrix to obtain zero coefficients ...
            # ... for the variable on the rows below
            if significant:
                # iterate over rows, ascendant
                mat = self.gauss_elimination_cols_ascendant(mat, col, m)

        # check the significant and zero rows of the matrix and ...
        # ... determine if the matrix can be solved
        solution, significant = self.check_significant()
        mat = self.matrix_significant

        # if solution is "infinitely many solutions" or "no solutions":
        if not significant:
            return f"{solution}"
        else:
            # gauss elimination - iterate descendant over variables columns
            for col in range(n - 2, -1, -1):
                # iterate over rows, descendant
                mat = self.gauss_elimination_cols_descendant(mat, col)

        if self.swap_list:
            self.swap_cols()
        self.solution_list = [mat[x, n - 1] for x in range(n - 1)]

        return f"The solution is {tuple(self.solution_list)}"


new_matrix = Matrix(matrix_array, rows_number, cols_number)
print(new_matrix.solution)

with open(args.outfile, 'w') as file:
    if not new_matrix.solution_list:
        file.write(new_matrix.solution)
    else:
        for el in new_matrix.solution_list:
            file.write(str(el) + '\n')

print(f"Saved to out.txt")

# CLI -> Terminal commands:
"""
python solver.py --infile in.txt --outfile out.txt
python solver.py --infile in1.txt --outfile out1.txt
python solver.py --infile in2.txt --outfile out2.txt
python solver.py --infile in3.txt --outfile out3.txt
python solver.py --infile in19.txt --outfile out19.txt
python solver.py --infile in24.txt --outfile out24.txt
"""
