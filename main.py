import retro


def print_actions(environment):
    for index in range(2 ** 9):
        action = list("{:08b}".format(index))
        print("%s %30s" % (action, environment.get_action_meaning(action)))


def main():
    environment = retro.make(game="DonkeyKong-nes")
    print_actions(environment)
    environment.reset()
    while True:
        environment.render()
        next_state, reward, done, information = environment.step(environment.action_space.sample())
        if done:
            break
    environment.close()


if __name__ == "__main__":
    main()
