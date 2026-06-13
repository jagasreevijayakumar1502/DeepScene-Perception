import random

TEMPLATES = [
    '{source} is {relation} {target}.',
    'The {source} appears to be {relation} the {target}.',
    'A {source} is positioned {relation} the {target}.',
]

RELATION_PHRASES = {
    'left_of': 'to the left of',
    'right_of': 'to the right of',
    'above': 'above',
    'below': 'below',
    'in_front_of': 'in front of',
    'behind': 'behind',
    'near': 'close to',
    'far': 'far from',
    'overlapping': 'overlapping with',
    'inside': 'inside',
    'approaching': 'approaching',
    'moving_away': 'moving away from',
    'crossing': 'crossing through',
    'stationary': 'stationary near',
    'close': 'close to'
}

def generate_description(objects, edges, reasoning):
    sentences = []
    
    if not objects:
        return ['No objects detected in the scene.']
    
    # Add primary object descriptions
    for obj in objects[:5]:
        conf = obj.get('confidence', 0.5)
        if conf > 0.4:
            sentences.append(f"Detected {obj['label']} at depth {obj.get('depth', 0):.1f} units.")
    
    # Add spatial relationships (limit to avoid repetition)
    unique_relations = set()
    for edge in edges[:6]:
        relation = RELATION_PHRASES.get(edge['relation'], edge['relation'])
        key = f"{edge['source']}-{edge['relation']}-{edge['target']}"
        if key not in unique_relations:
            template = random.choice(TEMPLATES)
            sentences.append(template.format(source=edge['source'], relation=relation, target=edge['target']))
            unique_relations.add(key)
    
    # Add scene context
    if reasoning and reasoning.get('layer4_scene_context'):
        sentences.append(reasoning['layer4_scene_context']['summary'])
    
    if not sentences:
        sentences.append('Scene analysis complete.')
    
    return sentences
