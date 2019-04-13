import xlrd
from cvxopt.modeling import variable, op
from sympy import *


def get_values(worksheet, row, column, string=False):
    values = []
    while worksheet.cell(row, column).value != xlrd.empty_cell.value:
        if string:
            values.append(str(worksheet.cell(row, column).value))
        else:
            values.append(worksheet.cell(row, column).value)
        row += 1
    return values


def linear_programming():
    workbook = xlrd.open_workbook('Zadachka.xlsx')
    worksheet = workbook.sheet_by_index(0)
    alpha = get_values(worksheet, 4, 1)
    beta = get_values(worksheet, 4, 2)
    v = get_values(worksheet, 4, 3)
    V = get_values(worksheet, 4, 4)
    teta = get_values(worksheet, 4, 5, string=True)
    F = worksheet.cell(0, 1).value
    T = worksheet.cell(1, 1).value
    k = [a/b for a, b in zip(V, v)]

    n = len(alpha)
    x = Symbol('x')
    teta_solved = [float(integrate(eval(teta[i]), (x, 0, 1))) for i in range(0, n)]
    x = variable(n, 'x')
    sum_var1 = 0
    sum_var2 = 0
    for i in range(0, n):
        sum_var1 += x[i] * v[i] * beta[i]

    for i in range(0, n):
        sum_var2 += x[i] * v[i] * alpha[i]

    z = -(sum_var1 - sum_var2)  # Функция цели "11.1"
    restrict1 = (sum_var2 <= F)  # "11.2"
    restrict2 = ([0 <= x[i] <= k[i] for i in range(0, n)])   # "11.3"
    restrict3 = ([x[i] * v[i] <= teta_solved[i] <= v[i] * (x[i] + 1) for i in range(0, n)])  # "11.4"
    x_non_negative = (x >= 0)

    problem = op(z, [restrict1, *restrict2, *restrict3, x_non_negative])
    problem.solve('glpk')
    print(problem.status)
    print("Максимум целевой функции:")
    print(abs(problem.objective.value()[0]))
    print("X:")
    print(x.value)


def main():
    print(linear_programming())


if __name__ == '__main__':
    main()
