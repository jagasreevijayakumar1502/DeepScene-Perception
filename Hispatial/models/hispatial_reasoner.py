import math


def analyze_scene(objects, edges):
    layer1 = _object_understanding(objects)
    layer2 = _pairwise_relationships(edges)
    layer3 = _group_interactions(objects, edges)
    layer4 = _scene_context(objects, edges)
    return {
        'layer1_object_understanding': layer1,
        'layer2_pairwise_spatial_relationships': layer2,
        'layer3_group_interaction_analysis': layer3,
        'layer4_scene_context': layer4,
    }


def _object_understanding(objects):
    summary = []
    for item in objects:
        summary.append({
            'object_id': item['object_id'],
            'label': item['label'],
            'position': item['position'],
            'size': item['size'],
            'depth': item['depth']
        })
    return summary


def _pairwise_relationships(edges):
    return [{'source': edge['source'], 'relation': edge['relation'], 'target': edge['target'], 'distance': edge['distance']} for edge in edges]


def _group_interactions(objects, edges):
    clusters = {}
    for obj in objects:
        clusters.setdefault(obj['label'], []).append(obj['object_id'])
    group_summary = [{'label': label, 'count': len(ids), 'members': ids} for label, ids in clusters.items() if len(ids) > 1]
    close_clusters = [edge for edge in edges if edge['relation'] in ('near', 'inside', 'overlapping')]
    return {
        'clusters': group_summary,
        'close_relationships': close_clusters[:8]
    }


def _scene_context(objects, edges):
    labels = [obj['label'] for obj in objects]
    if not labels:
        return {'summary': 'No scene objects detected.'}
    dominant = max(set(labels), key=labels.count)
    context = f"Detected {len(objects)} objects in the scene, dominated by {dominant}."
    if any(edge['relation'] == 'approaching' for edge in edges):
        context += ' The scene contains dynamic motion that suggests approach behavior.'
    return {
        'summary': context,
        'object_types': sorted(set(labels)),
        'edge_count': len(edges)
    }


def analyze_video_motion(previous_objects, current_objects):
    motions = []
    if not previous_objects or not current_objects:
        return motions
    
    for prev in previous_objects:
        best_target = None
        best_dist = float('inf')
        for cur in current_objects:
            dx = cur['position']['x'] - prev['position']['x']
            dy = cur['position']['y'] - prev['position']['y']
            dz = cur['position']['z'] - prev['position']['z']
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < best_dist:
                best_dist = dist
                best_target = cur
        
        if best_target is None or best_dist > 50:
            continue
        
        dx = best_target['position']['x'] - prev['position']['x']
        dz = best_target['position']['z'] - prev['position']['z']
        motion_dist = math.sqrt(dx*dx + dz*dz)
        
        if motion_dist < 2:
            continue
        
        if abs(dz) > abs(dx) * 1.5:
            if dz < 0:
                motions.append({'label': prev['label'], 'motion': 'approaching', 'distance': round(motion_dist, 2)})
            else:
                motions.append({'label': prev['label'], 'motion': 'moving_away', 'distance': round(motion_dist, 2)})
        else:
            if abs(dx) > 5:
                motions.append({'label': prev['label'], 'motion': 'crossing', 'distance': round(motion_dist, 2)})
            else:
                motions.append({'label': prev['label'], 'motion': 'stationary', 'distance': round(motion_dist, 2)})
    
    return motions
