# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features
    
        
class SmartFeatures(FeatureExtractor):
    """
    Safe upgrade: keeps SimpleExtractor behavior but adds strategic features
    that help Pacman perform better without breaking learning.
    """

    def getFeatures(self, state, action):
        from util import manhattanDistance

        features = util.Counter()
        features["bias"] = 1.0

        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        ghostStates = state.getGhostStates()
        capsules = state.getCapsules()

        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # --- 1. Original SimpleExtractor features ---
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts
        )
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0
        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        # --- 2. New Features ---

        # 2a. Closest Scared Ghost Distance (reciprocal)
        scaredGhosts = [g for i, g in enumerate(ghosts) if ghostStates[i].scaredTimer > 0]
        if scaredGhosts:
            minDist = min(manhattanDistance((next_x, next_y), g) for g in scaredGhosts)
            if minDist > 0:
                features["closest-scared-ghost"] = 1.0 / minDist

        # 2b. Escape Route Quality (number of legal moves)
        legalNeighbors = Actions.getLegalNeighbors((next_x, next_y), walls)
        features["escape-routes"] = len(legalNeighbors) / 4.0  # normalize to max 4

        # 2c. Food Density in Movement Direction (radius=3)
        radius = 3
        foodCount = 0
        for i in range(1, radius + 1):
            fx, fy = next_x + int(dx * i), next_y + int(dy * i)
            if 0 <= fx < food.width and 0 <= fy < food.height and not walls[fx][fy]:
                if food[fx][fy]:
                    foodCount += 1
        features["food-density"] = foodCount / float(radius)

        # 2d. Active Ghost Proximity Weighted (sum of reciprocal distances)
        activeGhosts = [g for i, g in enumerate(ghosts) if ghostStates[i].scaredTimer == 0]
        ghostProximity = 0.0
        for g in activeGhosts:
            dist = manhattanDistance((next_x, next_y), g)
            if dist > 0:
                ghostProximity += 1.0 / dist
        features["active-ghost-proximity"] = ghostProximity

        # 2e. Capsule Proximity (encourage heading toward power pellets)
        if capsules:
            minCapsuleDist = min(manhattanDistance((next_x, next_y), c) for c in capsules)
            features["capsule-proximity"] = 1.0 / (minCapsuleDist + 1.0)
        else:
            features["capsule-proximity"] = 0.0

        # --- Normalize all features ---
        features.divideAll(10.0)
        return features
