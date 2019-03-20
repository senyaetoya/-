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


def get_data(filepath, **coeffs):
    #  открываем нужный лист в выбранной в форме книге excel
    workbook = xlrd.open_workbook(str(filepath))
    worksheet = workbook.sheet_by_index(0)
    #  получаем массивы данных с листа по столбцам
    alpha = get_values(worksheet, 1, 1)
    beta = get_values(worksheet, 1, 2)
    v = get_values(worksheet, 1, 3)
    V = get_values(worksheet, 1, 4)
    teta = get_values(worksheet, 1, 5, string=True)
    k = [a / b for a, b in zip(V, v)]
    # убеждаемся, что массивы одинаковой длины
    assert len(teta) == len(v) == len(V) == len(alpha) == len(beta)
    #  приводим в нужный формат коэффициенты с формы
    T = int(coeffs['T'])
    F = float(coeffs['F'])
    # интегрируем тета
    teta_integrated = [float(integrate(eval(teta[i]), (Symbol('x'), 0, T))) for i in range(0, len(teta))]
    #  вызываем функцию-решатель
    return [alpha, beta, v, teta_integrated, k, F]


def solve_problem(alpha, beta, v, teta_integrated, k, F):
    n = len(alpha)
    problem = LpProblem('Zadachka', LpMaximize)
    x = LpVariable.dicts('x', range(n), lowBound=0, cat=LpInteger)
    sum_var1 = lpSum([x[i] * v[i] * beta[i] for i in range(0, n)])
    sum_var2 = lpSum([x[i] * v[i] * alpha[i] for i in range(0, n)])
    problem += sum_var1 - sum_var2  # 'Функция цели "11.1"'
    problem += sum_var2 <= F  # "11.2"
    constraint1 = [x[i] <= k[i] for i in range(0, n)]
    for cnstr in constraint1:
        problem += cnstr
    constraint2 = [x[i] * v[i] <= teta_integrated[i] for i in range(0, n)]
    for cnstr in constraint2:
        problem += cnstr
    constraint3 = [teta_integrated[i] <= v[i] * (x[i] + 1) for i in range(0, n)]
    for cnstr in constraint3:
        problem += cnstr
    problem.solve()
    return [problem.variables(), pulp.LpStatus[problem.status], \
            pulp.value(problem.objective), problem.solutionTime]


def show_results(variables, status, solution, time):
    xs = []
    for v in variables:
        xs.append(str(v.name) + " = " + str(v.varValue))
    status = 'Статус: ' + status
    solution = 'Значение целевой функции: ' + str(solution)
    time = 'Время решения: ' + str(time) + ' сек.'
    results = [status, solution, *xs, time]
    return results


def integer_lp(filepath, **coeffs):
    data = get_data(filepath, **coeffs)
    problem = solve_problem(*data)
    return show_results(*problem)


def main():
    print(get_data('Zadachka.xlsx', T=30, F=100000))


if __name__ == '__main__':
    main()
