from pulp import *
import openpyxl
from sympy import *


def get_values(worksheet, row, column, string=False):
    values = []
    while worksheet.cell(row=row, column=column).value:
        if string:
            values.append(str(worksheet.cell(row, column).value))
        else:
            values.append(worksheet.cell(row, column).value)
        row += 1
    return values


def get_data(filepath, **coeffs):
    #  открываем нужный лист в выбранной в форме книге excel
    workbook = openpyxl.load_workbook(filename=str(filepath))
    worksheet = workbook.worksheets[0]
    #  получаем массивы данных с листа по столбцам
    alpha = get_values(worksheet, 2, 2)
    beta = get_values(worksheet, 2, 3)
    v = get_values(worksheet, 2, 4)
    V = get_values(worksheet, 2, 5)
    teta = get_values(worksheet, 2, 6, string=True)
    k = [a / b for a, b in zip(V, v)]
    # убеждаемся, что массивы одинаковой длины
    assert len(teta) == len(v) == len(V) == len(alpha) == len(beta)
    #  приводим в нужный формат коэффициенты с формы
    T = int(coeffs['T'])
    F = float(coeffs['F'])
    sort_type = coeffs['sort']
    # интегрируем тета
    teta_integrated = [float(integrate(eval(teta[i]), (Symbol('x'), 0, T))) for i in range(0, len(teta))]
    #  вызываем функцию-решатель
    return [alpha, beta, v, teta_integrated, k, F], sort_type, workbook, worksheet


def sort_data(alpha, beta, v, teta_integrated, k, F, sort_type):
    print(teta_integrated, 'teta')
    print(alpha, 'alpha')
    print(beta, 'beta')
    print(k, 'k')
    print(v, 'v')
    if sort_type == 'β/α (max -> min)':
        ba = [b / a for a, b in zip(alpha, beta)]
        print(ba, 'b/a')
        zipped = list(zip(ba, teta_integrated, alpha, beta, k, v))
        zipped.sort(reverse=True)
        ba, teta_integrated, alpha, beta, k, v = zip(*zipped)
        print(ba, 'b/a')
    else:
        zipped = list(zip(teta_integrated, alpha, beta, k, v))
        zipped.sort(reverse=True)
        teta_integrated, alpha, beta, k, v = zip(*zipped)
    print(teta_integrated, 'teta')
    print(alpha, 'alpha')
    print(beta, 'beta')
    print(k, 'k')
    print(v, 'v')
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


def show_results(variables, status, solution, time, sort_type):
    xs = []
    for v in variables:
        xs.append(str(v.name) + " = " + str(v.varValue))
    status = 'Статус: ' + status
    solution = 'Значение целевой функции: ' + str(solution)
    time = 'Время решения: ' + str(time) + ' сек.'
    sort_type = 'Сортировка: ' + sort_type
    results = [status, sort_type, solution, *xs, time]
    return results


def write_to_excel(workbook, worksheet, filepath, sort_type, *problem):
    variables, status, solution, time = problem
    worksheet.cell(2, 8).value = 'Статус: ' + status
    worksheet.cell(3, 8).value = 'Сортировка: ' + sort_type
    worksheet.cell(4, 8).value = 'Значение целевой функции: ' + str(solution)
    for i in range(len(variables)):
        worksheet.cell(5 + i, 8).value = (str(variables[i].name) +
                                          " = " + str(variables[i].varValue))
    worksheet.cell(6 + i, 8).value = 'Время решения: ' + str(time) + ' сек.'
    workbook.save(filepath)


def integer_lp(filepath, **coeffs):
    data, sort_type, workbook, worksheet = get_data(filepath, **coeffs)
    sorted_data = sort_data(*data, sort_type)
    problem = solve_problem(*sorted_data)
    write_to_excel(workbook, worksheet, filepath, sort_type, *problem)
    return show_results(*problem, sort_type)


def main():
    print(integer_lp('Zadachka.xlsx', T=30, F=100000, sort='β/α (max -> min)'))


if __name__ == '__main__':
    main()
