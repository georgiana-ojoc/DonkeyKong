import retro


def main():
    movie = retro.Movie("movies/level1_2.bk2")
    movie.step()
    environment = retro.make(game=movie.get_game(), state=None, use_restricted_actions=retro.Actions.ALL,
                             players=movie.players)

    environment.initial_state = movie.get_state()
    environment.reset()

    last_state = None
    while movie.step():
        keys = []
        for player in range(movie.players):
            for index in range(environment.num_buttons):
                keys.append(movie.get_key(index, player))
        next_state, reward, done, information = environment.step(keys)
        environment.render()
        last_state = environment.em.get_state()
        if done:
            break

    if last_state is not None:
        environment.initial_state = last_state
        environment.reset()
        while True:
            next_state, reward, done, information = environment.step(environment.action_space.sample())
            environment.render()
            if done:
                break


if __name__ == "__main__":
    main()
