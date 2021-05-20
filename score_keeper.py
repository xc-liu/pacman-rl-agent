
last_score = 0
num_games = 0

def save_score_final(score):
    global last_score
    last_score = score

def return_score():
    global last_score
    return last_score


def save_number_games(number_games):
    global num_games
    num_games = number_games

def return_number_games():
    global num_games
    return num_games