from pulp import *
import openpyxl
from sympy import *
from math import ceil, floor


class Solved(object):
    def __init__(self, problem, number, func_value, vars_value,
                 continuos_var=False, cont_var_value=False):
        self.problem = problem
        self.number = number
        self.func_value = func_value
        self.cont_var = continuos_var
        self.vars_value = vars_value
        self.cont_var_value = cont_var_value

    def __repr__(self):
        return self.cont_var

    def __str__(self):
        return self.value

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


def copy_problem(problem, acc):  # бросил, делаю через var_values в классе Solved
    copy = LpProblem(acc, LpMaximize)
    n = len(problem.variables())
    y = LpVariable.dicts('y', range(n), lowBound=0, cat=LpContinuous)
    print(problem.constraints)
    for key in problem.constraints:
        coeffs = []
        constr = problem.constraints[key]
        constant = constr.constant
        for var in constr:
            coeffs.append(constr[var])

    copy.solve()
    print('kek')

    return problem


def solve_problem(problem):
    queue = []  # очередь на ветвление
    optimal = []  # лист оптимальных целочисленных решений
    acc = 0  # счетчик задач
    max_z = 0  # максимальное значение целевой функции
    problem_copy = problem.deepcopy()
    problem_copy.solve()

    if problem_copy.status != 1:
        return [problem_copy.variables(), problem_copy.status,
                pulp.value(problem_copy.objective), acc]  # возвращаем задачу, если она не оптимальна
    else:
        for v in problem_copy.variables():  # проверяем все х на целочисленность
            if v.varValue != int(v.varValue):  # если нецелочисленное, то добавляем в очередь на ветвление
                var_values = [var.varValue for var in problem_copy.variables()]
                queue.append(Solved(problem, acc, pulp.value(problem_copy.objective),
                                    var_values, v, v.varValue))  # добавляем первую проблему в очередь
                status, acc, solution = branch_and_bound(queue, max_z, acc, optimal)  # шаг 3
                return [status, solution.problem.variables(), solution.func_value, solution.vars_value,
                        solution.number, acc]
                # возвращаем проблему, если она целочислена, в главную функцию


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
            return 'Оптимально', acc, solution  # возвращаем успешное решение в формате Solved
        else:
            return 'unfeasible'  # тут тоже должен быть солюшн


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

        if child_solved.cont_var is False and max_z == 0:  # шаг 5
            max_z = child_solved.func_value
            optimal.append(child_solved)
            for prob in queue:  # проверяем, будем ли ветвить эту задачу
                if prob.value > max_z:
                    queue.append(child_solved)
                    break

        elif child_solved.cont_var is False and child_solved.value >= max_z:
            optimal.append(child_solved)

        elif child_solved.cont_var is True and max_z == 0:
            queue.append(child_solved)

    if i == 1:
        return vetv(parent_problem, acc, queue, max_z, optimal, i + 1)  # ветвим второй раз
    else:
        return queue, max_z, acc, optimal


def show_results(status, variables, func_value, vars_value, number_of_optimal, acc, sort_type):
    status = 'Статус: ' + status
    xs = [str(x[0]) + ' = ' + str(x[1]) for x in zip(variables, vars_value)]
    number_of_optimal = 'Номер оптимальной задачи: ' + str(number_of_optimal)
    func_value = 'Значение целевой функции: ' + str(func_value)
    sort_type = 'Сортировка: ' + sort_type
    acc = 'Кол-во решенных ЗЛП: ' + str((acc + 1))
    results = [status, func_value, *xs, sort_type, acc, number_of_optimal]
    return results


def integer_lp(filepath, **coeffs):
    data, sort_type, workbook, worksheet = get_data(filepath, **coeffs)
    sorted_data = sort_data(*data, sort_type)
    problem = form_problem(*sorted_data)
    solved_problem = solve_problem(problem)
    return show_results(*solved_problem, sort_type)


def main():
    answer = integer_lp('Zadachka.xlsx', T=30, F=100000, sort='b/a')
    print(*answer, sep="\n")


if __name__ == '__main__':
    main()
