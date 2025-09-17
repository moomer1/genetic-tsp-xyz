from typing import List, Tuple

points3D = Tuple[float, float, float]

def parse_input_file(path: str) -> List[points3D]:
    cities: List[points3D] = []
    with open(path, "r") as f:
        n = int(f.readline().strip())
        while len(cities) < n:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            x, y, z = map(float, line.split())
            cities.append((x,y,z))
    return cities

print(parse_input_file('input.txt'))