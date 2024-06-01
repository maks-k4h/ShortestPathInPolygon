import tkinter as tk
import random
import math
from typing import Tuple, List

import numpy as np

from main import Polygon, polygon_shortest_path


def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * np.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles


def clip(value, lower, upper):
    return min(upper, max(value, lower))


def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points


class PolygonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shortest Path in Polygon")

        self.canvas = tk.Canvas(root, width=600, height=400, bg="white")
        self.canvas.pack()

        self.save_polygon_button = tk.Button(root, text="Save Polygon", command=self.save_polygon)
        self.save_polygon_button.pack()

        self.process_button = tk.Button(root, text="Process", command=self.process)
        self.process_button.pack()

        self.reset_button = tk.Button(root, text="Reset", command=self.reset)
        self.reset_button.pack()

        self.save_polygon_button = tk.Button(root, text="Generate Polygon", command=self.generate_polygon)
        self.save_polygon_button.pack()

        self.vertices = []
        self.polygon_saved = False
        self.pt_start = None
        self.pt_end = None
        self.canvas.bind("<Button-1>", self.add_point)

    def add_point(self, event):
        x, y = event.x, event.y
        if not self.polygon_saved:
            self.vertices.append((x, y))
            self.render_vertices()
        else:
            if self.pt_start is None:
                self.pt_start = (x, y)
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="Blue")
            elif self.pt_end is None:
                self.pt_end = (x, y)
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="Yellow")
            else:
                return

    def save_polygon(self):
        if len(self.vertices) < 3:
            tk.messagebox.showwarning("Warning", "At least 3 vertices are needed to form a polygon.")
            return

        self.canvas.create_line(*self.vertices[-1], self.vertices[0], fill="black")
        self.polygon_saved = True

    def generate_polygon(self):
        self.vertices = generate_polygon((300, 200), 120, 0.5, 0.4, 70)
        self.polygon_saved = True
        self.pt_start = self.pt_end = None
        self.render_vertices()
        self.save_polygon()

    def render_vertices(self):
        self.canvas.delete("all")

        for i in range(len(self.vertices)):
            x, y = self.vertices[i]
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="black")
            self.canvas.create_text(x + 10, y + 10, text=f"{i}", fill='red')
            if i != 0:
                self.canvas.create_line(*self.vertices[i-1], x, y, fill="black")


    def reset(self):
        self.vertices = []
        self.polygon_saved = False
        self.pt_start = None
        self.pt_end = None
        self.canvas.delete('all')

    def process(self):
        if not self.polygon_saved:
            tk.messagebox.showwarning("Warning", "Draw and save the polygon")
            return
        if self.pt_start is None or self.pt_end is None:
            tk.messagebox.showwarning("Warning", "Draw start and end points")

        polygon = Polygon(np.asarray(self.vertices, dtype=np.float32))
        points = polygon_shortest_path(polygon, np.asarray(self.pt_start), np.asarray(self.pt_end))
        self.vertices = polygon.points.tolist()

        def get_point(j):
            if j == 0:
                p = self.pt_start
            elif j == 1:
                p = self.pt_end
            else:
                p = self.vertices[j - 2]
            return p

        for i1, i2 in zip(points, points[1:]):
            self.canvas.create_line(*get_point(i1), *get_point(i2), fill="green")


if __name__ == "__main__":
    root = tk.Tk()
    app = PolygonApp(root)
    root.mainloop()
