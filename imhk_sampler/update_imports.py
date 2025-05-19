
import sys
sys.path.insert(0, '.')

# Update imports to use v2 modules
with open('fixed_publication_results_v2.py', 'r') as f:
    content = f.read()

# Update import statements
content = content.replace('from fixed_samplers import', 'from fixed_samplers_v2 import')
content = content.replace('from fixed_tv_distance_calculation import', 'from fixed_tv_distance_calculation_v2 import')

with open('fixed_publication_results_v2.py', 'w') as f:
    f.write(content)

print("Imports updated successfully")
