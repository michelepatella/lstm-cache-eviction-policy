from utils.logs.levels.info_logger import info


def calculate_key_scores(num_keys, num_steps, prob_matrix, conf_matrix):
    scores = {}

    # for each key calculate a score
    for k in range(num_keys):
        score = 0.0
        for t in range(num_steps):
            # calculate the final score as a combination of
            # probability of a key of being used and CIs related
            # to that prediction
            score += alpha * prob_matrix[t, k] + beta * conf_matrix[t, k]
        scores[k] = score

    # normalize scores in [0,1]
    min_score = min(scores.values())
    max_score = max(scores.values())
    score_range = max_score - min_score if max_score != min_score else 1.0
    scores = {k: (v - min_score) / score_range for k, v in scores.items()}

    info("Key scores calculated.")

    return scores
