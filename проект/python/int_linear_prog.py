from pulp import *
import xlrd
from sympy import *

def get_values(worksheet, row, column, string=False):
    values = []
    while row < worksheet.nrows and worksheet.cell(row, column).value != xlrd.empty_cell.value:
        if string:
            values.append(str(worksheet.cell(row, column).value))
        else:
            values.append(worksheet.cell(row, column).value)
        row += 1
    return values


def linear_programming(filepath, **coeffs):
    workbook = xlrd.open_workbook(str(filepath))
    worksheet = workbook.sheet_by_index(0)
    alpha = get_values(worksheet, 1, 1)
    beta = get_values(worksheet, 1, 2)
    v = get_values(worksheet, 1, 3)
    V = get_values(worksheet, 1, 4)
    teta = get_values(worksheet, 1, 5, string=True)
    assert len(teta) == len(v) == len(V) == len(alpha) == len(beta)
    F = float(coeffs['F'])
    T = int(coeffs['T'])
    k = [a/b for a, b in zip(V, v)]
    n = len(alpha)
    x = Symbol('x')
    teta_solved = [float(integrate(eval(teta[i]), (x, 0, T))) for i in range(0, n)]

    problem = LpProblem('Zadachka', LpMaximize)
    x = LpVariable.dicts('x', range(n), lowBound=0, cat=LpInteger)
    sum_var1 = lpSum([x[i] * v[i] * beta[i] for i in range(0, n)])
    sum_var2 = lpSum([x[i] * v[i] * alpha[i] for i in range(0, n)])
    problem += sum_var1 - sum_var2  # 'Функция цели "11.1"'
    problem += sum_var2 <= F  # "11.2"
    constrain1 = [x[i] <= k[i] for i in range(0, n)]
    for cnstr in constrain1:
        problem += cnstr
    constrain2 = [x[i] * v[i] <= teta_solved[i] for i in range(0, n)]
    for cnstr in constrain2:
        problem += cnstr
    constrain3 = [teta_solved[i] <= v[i] * (x[i] + 1) for i in range(0, n)]
    for cnstr in constrain3:
        problem += cnstr

    problem.solve()

    xs = []
    for v in problem.variables():
        xs.append(str(v.name) + " = " + str(v.varValue))
    status = 'Статус: ' + pulp.LpStatus[problem.status]
    solution = 'Значение целевой функции: ' + str(pulp.value(problem.objective))
    time = 'Время решения: ' + str(problem.solutionTime) + ' сек.'
    results = [status, solution, *xs, time]
    problem.writeLP('Lp')
    problem.writeMPS('Mps')

    return results


def main(filepath, **coeffs):
    return linear_programming(filepath, **coeffs)


if __name__ == '__main__':
    main()
