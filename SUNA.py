import numpy as np
from collections import defaultdict


class Rectangle:
    def __init__(self, x_min, x_max, y_min, y_max, points):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.points = points  # List of (x,y) points in rectangle

    def min_distance_to_point(self, x, y):
        """Calculate minimum l∞ distance from point to rectangle"""
        dx = 0 if self.x_min <= x <= self.x_max else \
            min(abs(x - self.x_min), abs(x - self.x_max))
        dy = 0 if self.y_min <= y <= self.y_max else \
            min(abs(y - self.y_min), abs(y - self.y_max))
        return max(dx, dy)

    def max_distance_to_point(self, x, y):
        """Calculate maximum l∞ distance from point to rectangle"""
        dx = max(abs(x - self.x_min), abs(x - self.x_max))
        dy = max(abs(y - self.y_min), abs(y - self.y_max))
        return max(dx, dy)


def calculate_radius_suna(R, T, k, join_key):
    """
    Calculate k-nearest neighbor radius following SUNA milestone 1.2.1 exactly
    
    1. Form rectangles for each join key
    2. Find k-closest rectangles by min distance (may be overestimate)
    3. Estimate radius using only points in selected rectangles
    """
    # Form rectangles for each join key
    rectangles = {}
    for key in set(R[join_key].unique()) & set(T[join_key].unique()):
        x_vals = R[R[join_key] == key]['x']
        y_vals = T[T[join_key] == key]['y']

        points = []
        for x in x_vals:
            for y in y_vals:
                points.append((x, y))

        rectangles[key] = Rectangle(
            x_min=x_vals.min(),
            x_max=x_vals.max(),
            y_min=y_vals.min(),
            y_max=y_vals.max(),
            points=points
        )

    radii = []

    # For each point in the hypothetical join
    for key1, rect1 in rectangles.items():
        for x, y in rect1.points:
            # Find k-closest rectangles by minimum distance
            rect_distances = []
            for key2, rect2 in rectangles.items():
                if key1 != key2:
                    min_dist = rect2.min_distance_to_point(x, y)
                    max_dist = rect2.max_distance_to_point(x, y)
                    rect_distances.append((min_dist, max_dist, key2))

            # Sort by minimum distance
            rect_distances.sort(key=lambda x: x[0])

            # Take enough rectangles to guarantee k neighbors
            # (may be overestimate as per milestone requirements)
            selected_rects = []
            total_points = 0
            for min_dist, max_dist, rect_key in rect_distances:
                selected_rects.append((min_dist, max_dist, rect_key))
                total_points += len(rectangles[rect_key].points)
                if total_points >= k:
                    break

            # Estimate radius using selected rectangles
            if not selected_rects:
                # Only points in same rectangle
                max_dist = rect1.max_distance_to_point(x, y)
                radii.append(max_dist / 2)
            else:
                # Use maximum possible distance to k-th closest rectangle
                # This maintains the overestimate property required by milestone
                _, max_dist, _ = selected_rects[-1]
                radii.append(max_dist / 2)

    return np.array(radii)
