import numpy as np
from typing import NamedTuple, List, Tuple, Generator


class Polygon:
    def __init__(self, points: np.ndarray):
        assert len(points.shape) == 2 and points.shape[1] == 2, points.shape
        self.points = points

    @property
    def edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(self.points[i], self.points[(i + 1) % len(self.points)]) for i in range(len(self.points))]


def caconical_angle(v):
    a = np.arctan2(*v)
    return a if a >= 0 else a + 2 * np.pi


def get_visible_points(point: np.ndarray, polygon: Polygon) -> np.ndarray:
    # return distance to the point or -1 if it is not visible

    # Construct edge intersection table
    intersections = {}
    for p in polygon.points:
        angle = caconical_angle(p - point)
        intersections[angle] = []

    for i, (p_s, p_e) in enumerate(polygon.edges):
        a_s = caconical_angle(p_s - point)
        a_e = caconical_angle(p_e - point)
        if a_s > a_e:
            a_s, a_e = a_e, a_s
        if a_e - a_s == np.pi:
            continue  # query point on this edge
        for a in intersections.keys():
            if (a_e - a_s < np.pi) and not (a_s <= a <= a_e):
                continue
            if (a_e - a_s > np.pi) and not (a_e <= a or a <= a_s):
                continue
            intersections[a].append(i)

    # Calculate visibilities
    visibilities = []
    for i, p in enumerate(polygon.points):
        angle = caconical_angle(p - point)
        edges_idx = intersections[angle]
        obstructed = False
        for edge_idx in edges_idx:
            if i == edges_idx or i == (edge_idx + 1) % len(polygon.edges):
                continue  # edge containing query point
            # determine if the edge obstructs visibility of p
            # i.e., determine if the two points are located on different sides
            edge_p1, edge_p2 = polygon.edges[edge_idx]
            edge_normal = (edge_p1 - edge_p2) @ np.array([[0, -1], [1, 0]])
            if np.dot(edge_normal, p - edge_p1) * np.dot(edge_normal, point - edge_p1) < 0:
                # does obstruct
                visibilities.append(-1)
                obstructed = True
                break
        if not obstructed:
            # no obstructions, compute distance
            visibilities.append(np.linalg.norm(p - point))

    return np.asarray(visibilities)




def main() -> None:
    polygon = Polygon(np.array([(1, -2), (2, 0), (1, 2), (0, 3), (4, 3), (4, 0), (5, 0), (5, -1)]))
    point = (2, -1)
    print(get_visible_points(point, polygon))


if __name__ == '__main__':
    main()
