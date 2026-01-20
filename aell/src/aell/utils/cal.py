def cal(data):
    total_objective = 0
    count = 0

    for key, value in data.items():
        objective_value = value.get("objective", None)
        if objective_value is not None:
            total_objective += objective_value
            count += 1

    mean_objective = total_objective / count if count > 0 else None
    return mean_objective