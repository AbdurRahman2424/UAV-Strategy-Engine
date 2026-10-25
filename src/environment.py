"""Seeded terrain maps, coordinate conventions, and connected components."""
from collections import deque
import numpy as np
from config import EnvironmentConfig


class Environment:
    """Coordinates are (x, y); every array is indexed [y, x]."""

    def __init__(self, config=EnvironmentConfig()):
        self.config = config
        self.size = n = config.size
        rng = np.random.default_rng(config.seed)
        yy, xx = np.mgrid[:n, :n]
        self.priority = np.ones((n, n), dtype=float)
        for cx, cy, radius, value in ((.25, .28, .15, 5), (.74, .69, .16, 5),
                                      (.68, .24, .17, 3), (.28, .73, .18, 3)):
            self.priority[(xx-cx*n)**2+(yy-cy*n)**2 < (radius*n)**2] = value
        self.no_fly = np.zeros((n, n), dtype=bool)
        self.no_fly[int(.35*n):int(.58*n), int(.43*n):int(.54*n)] = True
        self.no_fly[int(.08*n):int(.20*n), int(.77*n):int(.90*n)] = True
        self.obstacles = np.zeros((n, n), dtype=bool)
        for _ in range(max(8, n//4)):
            x, y = rng.integers(1, n-4, size=2)
            w, h = rng.integers(2, max(3, n//11)+1, size=2)
            self.obstacles[y:y+h, x:x+w] = True
        self.traversable = ~(self.no_fly | self.obstacles)
        self.hazard = np.zeros((n, n))
        for cx, cy, sigma, strength in ((.29, .35, .12, .9), (.70, .65, .13, 1),
                                         (.56, .13, .09, .6)):
            self.hazard += strength*np.exp(-((xx-cx*n)**2+(yy-cy*n)**2)/(2*(sigma*n)**2))
        self.hazard = np.clip(self.hazard, 0, 1)
        self.target_probability = self.priority * (.6 + .4*rng.random((n, n)))
        self.target_probability *= self.traversable
        self.target_probability /= self.target_probability.sum()
        cells = rng.choice(n*n, min(config.target_count, self.traversable.sum()),
                           replace=False, p=self.target_probability.ravel())
        self.targets = np.zeros((n, n), dtype=bool)
        self.targets.ravel()[cells] = True
        self.coverage = np.zeros((n, n), dtype=bool)
        self.rebuild_navigation()

    def rebuild_navigation(self):
        """Call after editing terrain (useful for tests and custom scenarios)."""
        n = self.size
        self.valid_cells = np.column_stack(np.nonzero(self.traversable)[::-1])
        if not len(self.valid_cells):
            raise ValueError('The environment has no traversable deployment cells.')
        self.components = np.full((n, n), -1, dtype=int)
        self.component_cells = []
        for x, y in self.valid_cells:
            if self.components[y, x] >= 0:
                continue
            index = len(self.component_cells)
            queue = deque([(int(x), int(y))])
            self.components[y, x] = index
            cells = []
            while queue:
                a, b = queue.popleft()
                cells.append((a, b))
                for c, d in ((a+1,b),(a-1,b),(a,b+1),(a,b-1)):
                    if 0 <= c < n and 0 <= d < n and self.traversable[d,c] and self.components[d,c] < 0:
                        self.components[d,c] = index
                        queue.append((c,d))
            self.component_cells.append(np.array(cells, dtype=int))

    def valid(self, point):
        x, y = point
        return (isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer))
                and 0 <= x < self.size and 0 <= y < self.size and bool(self.traversable[y,x]))

    def nearest_valid(self, point, component=None):
        if self.valid(point) and (component is None or self.components[point[1],point[0]] == component):
            return tuple(map(int,point))
        cells = self.valid_cells if component is None else self.component_cells[component]
        p = np.nan_to_num(np.asarray(point, dtype=float), nan=0, posinf=self.size-1, neginf=0)
        return tuple(map(int, cells[np.argmin(np.sum((cells-p)**2, axis=1))]))

    def save(self, path):
        np.savez_compressed(path, priority=self.priority, hazard=self.hazard,
                            traversable=self.traversable, no_fly=self.no_fly,
                            obstacles=self.obstacles, targets=self.targets,
                            target_probability=self.target_probability, coverage=self.coverage)
