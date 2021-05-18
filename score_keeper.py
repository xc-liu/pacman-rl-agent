
last_score = 0

def save_score_final(score):
    global last_score
    last_score = score

def return_score():
    global last_score
    return last_score