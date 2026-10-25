"""Four-neighbour, hazard-aware A* with a bounded route cache."""
import heapq
from collections import OrderedDict


class AStarPlanner:
    def __init__(self, environment, hazard_weight=1.8, cache_size=40000):
        self.env = environment
        self.hazard_weight = hazard_weight
        self.cache_size = cache_size
        self.cache = OrderedDict()
        n = environment.size
        # Symmetric edge costs allow reverse-route cache reuse.
        self.neighbors = [[] for _ in range(n*n)]
        h = environment.hazard.ravel()
        for x, y in environment.valid_cells:
            i = int(y*n+x)
            for a, b in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if environment.valid((a,b)):
                    j = int(b*n+a)
                    self.neighbors[i].append((j, 1+hazard_weight*(h[i]+h[j])/2))

    def route(self, start, goal):
        if not self.env.valid(start) or not self.env.valid(goal):
            return None
        if self.env.components[start[1],start[0]] != self.env.components[goal[1],goal[0]]:
            return None
        key = (tuple(start), tuple(goal))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        reverse = (key[1],key[0])
        if reverse in self.cache:
            self.cache.move_to_end(reverse)
            return self.cache[reverse][::-1]
        n = self.env.size
        source, target = start[1]*n+start[0], goal[1]*n+goal[0]
        heap = [(0., 0., source)]
        scores = {source: 0.}
        parent = {}
        while heap:
            _, g, current = heapq.heappop(heap)
            if g > scores[current]:
                continue
            if current == target:
                path = [current]
                while path[-1] != source:
                    path.append(parent[path[-1]])
                result = tuple((int(i%n),int(i//n)) for i in reversed(path))
                self.cache[key] = result
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
                return result
            for nxt, cost in self.neighbors[current]:
                candidate = g+cost
                if candidate < scores.get(nxt, float('inf')):
                    scores[nxt] = candidate
                    parent[nxt] = current
                    heuristic = abs(nxt%n-goal[0])+abs(nxt//n-goal[1])
                    heapq.heappush(heap, (candidate+heuristic, candidate, nxt))
        return None
