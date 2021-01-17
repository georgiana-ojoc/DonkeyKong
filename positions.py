def get_positions(situation):
    with open("positions\\" + situation + ".txt") as file:
        content = file.read().split('\n')
    coordinates = []
    for line in content:
        line = line.split()
        coordinates += [{'x': int(line[0]), 'y': int(line[1])}]
    return coordinates


def get_all_positions():
    positions = {}
    situation = "ladders"
    positions[situation] = get_positions(situation)
    situation = "broken_ladders"
    positions[situation] = get_positions(situation)
    situation = "hammers"
    positions[situation] = get_positions(situation)
    situation = "start"
    positions[situation] = get_positions(situation)
    situation = "finish"
    positions[situation] = get_positions(situation)
    return positions


if __name__ == '__main__':
    print(get_all_positions())
