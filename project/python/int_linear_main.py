from pulp import *
import openpyxl
from sympy import *
from math import ceil, floor


class Solved(object):
    def __init__(self, problem, number, func_value, vars_value,
                 continuos_var=None, cont_var_value=None):
        self.problem = problem
        self.number = number
        self.func_value = func_value
        self.vars_value = vars_value
        self.cont_var = continuos_var
        self.cont_var_value = cont_var_value


class Solution(object):
    def __init__(self, status, acc, func_value=None, number=None,
                 variables=None, vars_value=None):
        self.status = status
        self.variables = variables
        self.func_value = func_value
        self.vars_value = vars_value
        self.number = number
        self.acc = acc


def get_values(worksheet, row, column, expression=False):
    values = []
    while worksheet.cell(row=row, column=column).value:
        if expression:
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
    teta = get_values(worksheet, 2, 6, expression=True)
    k = [a / b for a, b in zip(V, v)]
    # убеждаемся, что массивы одинаковой длины
    assert len(teta) == len(v) == len(V) == len(alpha) == len(beta)
    #  приводим в нужный формат коэффициенты с формы
    T = int(coeffs['T'])
    F = float(coeffs['F'])
    sort_type = coeffs['sort']
    # интегрируем тета
    x = Symbol('x')
    teta_integrated = [float(integrate(eval(teta[i]), (x, 0, T))) for i in range(0, len(teta))]
    #  вызываем функцию-решатель
    return [alpha, beta, v, teta_integrated, k, F], sort_type, workbook, worksheet


def sort_data(alpha, beta, v, teta_integrated, k, F, sort_type):
    if sort_type == 'b/a':
        ba = [b / a for a, b in zip(alpha, beta)]
        zipped = list(zip(ba, teta_integrated, alpha, beta, k, v))
        zipped.sort(reverse=True)
        ba, teta_integrated, alpha, beta, k, v = zip(*zipped)
    else:
        zipped = list(zip(teta_integrated, alpha, beta, k, v))
        zipped.sort(reverse=True)
        teta_integrated, alpha, beta, k, v = zip(*zipped)
    return [alpha, beta, v, teta_integrated, k, F]


def form_problem(alpha, beta, v, teta_integrated, k, F):
    n = len(alpha)
    problem = LpProblem('Zadachka', LpMaximize)
    x = LpVariable.dicts('x', range(n), lowBound=0, cat=LpContinuous)
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
    return problem


def solve_problem(problem):
    queue = []  # очередь на ветвление
    optimal = []  # лист оптимальных целочисленных решений
    acc = 1  # счетчик задач
    max_z = 0  # максимальное значение целевой функции
    problem_copy = problem.deepcopy()
    problem_copy.solve() # решаем первую проблему

    # возвращаем задачу, если она не оптимальна
    if problem_copy.status != 1:
        return Solution(status='Нерешаемо', acc=acc)
    else:
        # проверяем все х на целочисленность
        for v in problem_copy.variables():
            if v.varValue != int(v.varValue):
                var_values = [var.varValue for var in problem_copy.variables()]
                # добавляем первую проблему в очередь на ветвление
                queue.append(Solved(problem, acc, pulp.value(problem_copy.objective),
                                    var_values, v, v.varValue))
                status, acc, solution = branch_and_bound(queue, max_z, acc, optimal)
                # возвращаем проблему после branch_and_bound
                if solution:
                    return Solution(variables=solution.variables,
                                    vars_value=solution.vars_value,
                                    func_value=solution.func_value,
                                    number=solution.number,
                                    acc=acc,
                                    status=status)
                # если нет решений
                else:
                    return Solution(status=status, acc=acc)
            # если проблема целочислена на первом шаге
            else:
                return Solution(variables=problem_copy.variables(),
                                vars_value=[var.varValue for var in problem_copy.variables()],
                                func_value=pulp.value(problem_copy.objective),
                                number=acc,
                                acc=acc,
                                status='Оптимально')


def branch_and_bound(queue, max_z, acc, optimal):  # передаем сюда задачу на ветвление, в том числе нецелую переменную
    if queue:
        max_prob = queue[0]
        for prob in queue[1:]:
            if prob.value > max_prob.value:
                max_prob = prob
        queue.remove(max_prob)
        i = 1
        queue, max_z, acc, optimal = vetv(max_prob, acc, queue, max_z, optimal, i)
        return branch_and_bound(queue, max_z, acc, optimal)
    else:
        if optimal:
            solution = optimal[0]
            for prob in optimal[1:]:
                if prob.func_value > solution.func_value:
                    solution = prob
            return 'Оптимально', acc, solution   # возвращаем успешное решение в формате Solved
        else:
            return 'Нерешаемо', acc, False  # тут тоже должен быть солюшн


def vetv(parent_problem, acc, queue, max_z, optimal, i):
    child_problem = parent_problem.problem.deepcopy()
    if i == 1:
        child_problem += parent_problem.cont_var <= floor(parent_problem.cont_var_value)  # левая ветвь
    if i == 2:
        child_problem += parent_problem.cont_var >= ceil(parent_problem.cont_var_value)  # правая ветвь
    child_problem_copy = child_problem.deepcopy()
    child_problem_copy.solve()
    acc += 1

    if child_problem_copy.status == 1:  # формируем подпроблему в очередь
        var_values = [var.varValue for var in child_problem_copy.variables()]
        child_solved = Solved(child_problem, acc, pulp.value(child_problem_copy.objective),
                              var_values)
        for v in child_problem_copy.variables():  # ищем нецелочисленную переменную
            if v.varValue != int(v.varValue):
                child_solved.cont_var = v
                child_solved.cont_var_value = v.varValue
                break

        if child_solved.cont_var is None and max_z == 0:  # шаг 5
            max_z = child_solved.func_value
            optimal.append(child_solved)
            for prob in queue:  # проверяем, будем ли ветвить эту задачу
                if prob.value > max_z:
                    queue.append(child_solved)
                    break

        elif child_solved.cont_var is None and child_solved.func_value >= max_z:
            optimal.append(child_solved)

        elif child_solved.cont_var is not None and max_z == 0:
            queue.append(child_solved)

    if i == 1:
        return vetv(parent_problem, acc, queue, max_z, optimal, i + 1)  # ветвим второй раз
    else:
        return queue, max_z, acc, optimal


def show_results(solved, sort_type):
    status = 'Статус: ' + solved.status

    xs = [str(x[0]) + ' = ' + str(x[1]) for x in zip(solved.variables, solved.vars_value)]
    number_of_optimal = 'Номер оптимальной задачи: ' + str(solved.number)
    func_value = 'Значение целевой функции: ' + str(solved.func_value)
    sort_type = 'Сортировка: ' + sort_type
    acc = 'Кол-во решенных ЗЛП: ' + str(solved.acc)
    results = [status, func_value, *xs, sort_type, acc, number_of_optimal]
    return results


def integer_lp(filepath, **coeffs):
    data, sort_type, workbook, worksheet = get_data(filepath, **coeffs)
    sorted_data = sort_data(*data, sort_type)
    problem = form_problem(*sorted_data)
    solved_problem = solve_problem(problem)
    return show_results(solved_problem, sort_type)


def main():
    answer = integer_lp('Zadachka.xlsx', T=30, F=100000, sort='b/a')
    print(*answer, sep="\n")


if __name__ == '__main__':
    main()
