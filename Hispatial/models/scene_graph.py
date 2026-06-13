import math

RELATION_DISTANCE_NEAR = 120.0
RELATION_DISTANCE_FAR = 260.0


def _vector(a, b):
    return b['position']['x'] - a['position']['x'], b['position']['y'] - a['position']['y'], b['position']['z'] - a['position']['z']


def _distance(a, b):
    dx, dy, dz = _vector(a, b)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _directional_relations(a, b):
    dx, dy, dz = _vector(a, b)
    relations = []
    if abs(dx) > 20:
        relations.append('right_of' if dx > 0 else 'left_of')
    if abs(dy) > 20:
        relations.append('below' if dy > 0 else 'above')
    if abs(dz) > 3:
        relations.append('in_front_of' if dz < 0 else 'behind')
    return relations


def _distance_relation(dist):
    if dist < RELATION_DISTANCE_NEAR:
        return 'near'
    if dist > RELATION_DISTANCE_FAR:
        return 'far'
    return 'close'


def build_scene_graph(objects):
    edges = []
    if len(objects) < 2:
        return edges
    
    for idx_a, source in enumerate(objects):
        for idx_b, target in enumerate(objects):
            if idx_a >= idx_b:  
                continue
            dist = _distance(source, target)
            relations = _directional_relations(source, target)
            distance_relation = _distance_relation(dist)
            if distance_relation and distance_relation != 'close':
                relations.append(distance_relation)
            if source['bounding_box'] and target['bounding_box']:
                if _is_overlapping(source['bounding_box'], target['bounding_box']):
                    relations.append('overlapping')
            if not relations:
                relations.append('near')
            
            edges.append({
                'source': source['label'],
                'target': target['label'],
                'relation': relations[0],
                'distance': round(dist, 2),
                'details': relations
            })
    return edges


def _is_overlapping(box_a, box_b):
    ax, ay, aw, ah = box_a['x'], box_a['y'], box_a['width'], box_a['height']
    bx, by, bw, bh = box_b['x'], box_b['y'], box_b['width'], box_b['height']
    overlap_x = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_y = max(0, min(ay + ah, by + bh) - max(ay, by))
    return overlap_x > 0 and overlap_y > 0
