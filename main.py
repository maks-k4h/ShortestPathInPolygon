import random

import numpy as np
from typing import NamedTuple, List, Tuple, Generator


class Polygon:
    def __init__(self, points: np.ndarray):
        # todo: points have to be in clockwise order
        assert len(points.shape) == 2 and points.shape[1] == 2, points.shape
        self.points = points
        self._orient_polygon()

    def _orient_polygon(self, clockwise=True):
        e = np.asarray(self.edges)
        rightmost_i = np.argmax(e[:, 1] + e[:, 0], axis=0)[0]
        edge = self.edges[rightmost_i]

        print((edge[1] - edge[0])[1] < 0)
        if ((edge[1] - edge[0])[1] < 0) == clockwise:
            self.points = self.points[::-1]


    @property
    def edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(self.points[i], self.points[(i + 1) % len(self.points)]) for i in range(len(self.points))]


def _caconical_angle(v):
    a = np.arctan2(*v)
    return a if a >= 0 else a + 2 * np.pi


def _get_visible_vertices(point: np.ndarray, polygon: Polygon) -> np.ndarray:
    # return distance to the point or -1 if it is not visible

    point_in_polygon_idx = None  # such points need additional processing

    # Construct edge intersection table
    intersections = {}
    for i, p in enumerate(polygon.points):
        if np.linalg.norm(p - point) == 0:
            point_in_polygon_idx = i
        angle = _caconical_angle(p - point)
        intersections[angle] = []

    if point_in_polygon_idx is not None:
        # restrict possible angles from point
        i_in = point_in_polygon_idx - 1 if point_in_polygon_idx > 0 else len(polygon.edges) - 1
        i_out = point_in_polygon_idx
        restriction_n1 = polygon.edges[i_in][0] - polygon.edges[i_in][1]
        restriction_n2 = polygon.edges[i_out][1] - polygon.edges[i_out][0]

    for i, (p_s, p_e) in enumerate(polygon.edges):
        a_s = _caconical_angle(p_s - point)
        a_e = _caconical_angle(p_e - point)
        if a_s > a_e:
            a_s, a_e = a_e, a_s
        if a_e - a_s == np.pi:
            continue  # query point on this edge
        for a in intersections.keys():
            # todo: criterion can be improved (edge cases)
            if (a_e - a_s < np.pi) and not (a_s <= a <= a_e):
                continue
            if (a_e - a_s > np.pi) and not (a_e <= a or a <= a_s):
                continue
            intersections[a].append(i)

    # Calculate visibilities
    visibilities = []
    for i, p in enumerate(polygon.points):
        if point_in_polygon_idx is not None and point_in_polygon_idx == i:
            visibilities.append(0)
            continue
        obstructed = False
        angle = _caconical_angle(p - point)
        if point_in_polygon_idx is not None:
            a_a = _caconical_angle(p - point)
            a_v1 = _caconical_angle(restriction_n1)
            a_v2 = _caconical_angle(restriction_n2)
            if ((a_v1 <= a_v2) and not (a_v1 <= a_a <= a_v2) or
                    (a_v1 > a_v2 and not (a_a <= a_v2 or a_a >= a_v1))):
                obstructed = True
            pass
        edges_idx = intersections[angle]
        if not obstructed:
            for edge_idx in edges_idx:
                edge_p1, edge_p2 = polygon.edges[edge_idx]
                edge_normal = np.array([[0, -1], [1, 0]]) @ (edge_p1 - edge_p2)
                if i == edges_idx or i == (edge_idx + 1) % len(polygon.edges):
                    continue  # edge containing query point
                # determine if the edge obstructs visibility of p
                # i.e., determine if the two points are located on different sides
                if np.dot(edge_normal, p - edge_p1) * np.dot(edge_normal, point - edge_p1) < 0:
                    # does obstruct
                    obstructed = True
                    break
        if obstructed:
            visibilities.append(-1)
        else:
            # no obstructions, compute distance
            visibilities.append(np.linalg.norm(p - point))

    return np.asarray(visibilities)


def _are_directly_visible(point: np.ndarray, target: np.ndarray, polygon: Polygon) -> bool:
    for edge_p1, edge_p2 in polygon.edges:
        edge_normal = (edge_p1 - edge_p2) @ np.array([[0, -1], [1, 0]])
        if np.dot(edge_normal, point - edge_p1) * np.dot(edge_normal, target - edge_p1) < 0:
            return False
    return True


def _construct_visibility_graph(p: np.ndarray, q: np.ndarray, polygon: Polygon) -> np.array:
    # construct a connectivity matrix (n+2)x(n+2) where n — number of vertices in polygon
    # M[2:, 2:] describe visibility in polygon
    # The first row/column describes connectivity of p to the rest
    # The second row/column describes connectivity of q to the rest
    # value -1 — not visible, value >=0 — distance between visible points

    M = np.zeros((len(polygon.points) + 2, ) * 2)
    M[0, 1] = M[1, 0] = np.linalg.norm(p - q) if _are_directly_visible(p, q, polygon) else -1

    M[2:, 0] = M[0, 2:] = _get_visible_vertices(p, polygon)
    M[2:, 1] = M[1, 2:] = _get_visible_vertices(q, polygon)

    # visibilities between polygon vertices
    for i in range(0, len(polygon.points)):
        M[2 + i, 2:] = M[2:, 2 + i] = _get_visible_vertices(polygon.points[i], polygon)

    return M


def _run_dijkstra(M: np.ndarray, memo_checked, memo_distance, memo_path, v: int):
    if memo_checked[v] == 1:
        return
    curr_d = memo_distance[v]
    curr_p = memo_path[v]
    for i, d in enumerate(M[v]):
        if d < 0:
            continue
        if curr_d + d < memo_distance[i]:
            memo_distance[i] = curr_d + d
            memo_path[i] = curr_p + (i,)
    memo_checked[v] = 1
    for i in np.argsort(M[v]):
        if M[v][i] < 0:
            continue
        if memo_checked[i] == 1:
            continue
        _run_dijkstra(M, memo_checked, memo_distance, memo_path, i)


def _find_shortest(M: np.ndarray) -> List[int]:
    # find the shortest path from first vertex to the second given connectivity matrix
    # Provide costs; if vertices are not connected, set cost negative.
    memo_checked = np.zeros((M.shape[0]))
    memo_distance = np.zeros((M.shape[0])) + np.inf
    memo_path = [None] * M.shape[0]
    memo_distance[0] = 0
    memo_path[0] = (0, )
    _run_dijkstra(M, memo_checked, memo_distance, memo_path, 0)
    return list(memo_path[1])


def polygon_shortest_path(polygon: Polygon, start: np.ndarray, end: np.ndarray) -> List[int]:
    M = _construct_visibility_graph(start, end, polygon)
    return _find_shortest(M)


def main() -> None:
    polygon = Polygon(np.array([(1, -2), (2, 0), (1, 2), (0, 3), (4, 3), (4, 0), (5, 0), (5, -1)]))
    p = np.array((2, -1))
    q = np.array((1, 2.5))
    M = _construct_visibility_graph(p, q, polygon)
    for r in M:
        for e in r:
            print('{f:.2f}'.format(f=e), end='\t')
        print()
    print(_find_shortest(M))


if __name__ == '__main__':
    main()
