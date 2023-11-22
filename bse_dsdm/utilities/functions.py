import itertools

def get_combinations(list_of_params):
   combination = [] # empty list 
   for r in range(1, len(list_of_params) + 1):
      # to generate combination
      combination.extend(itertools.combinations(list_of_params, r))
   return combination