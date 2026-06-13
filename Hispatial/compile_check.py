import py_compile
import traceback
import sys

files = [
    'app.py',
    'models/sam2.py',
    'models/classifier.py',
    'models/depth_estimation.py',
    'models/scene_graph.py',
    'models/hispatial_reasoner.py',
    'models/language_generator.py'
]

status = 0
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f + ' OK')
    except Exception as e:
        print(f + ' ERROR:', e)
        traceback.print_exc()
        status = 1

sys.exit(status)
