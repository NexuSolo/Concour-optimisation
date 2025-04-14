import json
import math

SUBMISSION_FILE = 'solutions/1_example_submission.json'
DATASET_FILE = 'datasets/1_example.json'

def getSolutionScore(solution_txt, dataset_txt):
    """Evaluate the solution and return a tuple (score, is_valid, message).

    Arguments:;,
    solution -- the solution to be evaluated
    dataset -- the dataset for which the solution is made
    """
    try:
        dataset = json.loads(dataset_txt)
    except:
        return 0, False, 'Error while processing the dataset. Please contact the contest organizer.'
    
    dataset_nodes = set(node['id'] for node in dataset['intersections'])

    try:
        solution = json.loads(solution_txt)
    except:
        return 0, False, 'Submission is not a valid JSON object'
    
    if 'chargeStationId' not in solution:
        return 0, False, 'Submission does not contain the "chargeStationId" key'
    if not isinstance(solution['chargeStationId'], int):
        return 0, False, 'chargeStationId is not an integer'
    if solution['chargeStationId'] not in dataset_nodes:
        return 0, False, 'The charge station ID is not a valid intersection ID'
    
    if 'itinerary' not in solution:
        return 0, False, 'Submission does not contain the "itinerary" key'
    if not isinstance(solution['itinerary'], list):
        return 0, False, 'itinerary is not a list'
    if not solution['itinerary']:
        return 0, False, 'itinerary is empty'
    for node in solution['itinerary']:
        if not isinstance(node, int):
            return 0, False, f'itinerary contains a non-integer value "{node}"'
        if node not in dataset_nodes:
            return 0, False, f'itinerary contains an invalid intersection ID {node}'
    
    base_id = solution['chargeStationId']
    itinerary = solution['itinerary']
    if itinerary[0] != base_id:
        return 0, False, 'The first node in the itinerary should be the charge station'
    
    edge_length = {}
    max_possible_coverage = 0
    for road in dataset['roads']:
        max_possible_coverage += road['length']
        edge_length[(road['intersectionId1'], road['intersectionId2'])] = road['length']
        if not road['isOneWay']:
            edge_length[(road['intersectionId2'], road['intersectionId1'])] = road['length']

    visited_edges = set()
    total_covered_length = 0
    remaining_battery = dataset['batteryCapacity']
    current_day = 0
    warning_message = None
    for n1, n2 in zip(itinerary, itinerary[1:]):
        if n1 == base_id:
            current_day += 1
            remaining_battery = dataset['batteryCapacity']
        
        if (n1, n2) not in edge_length:
            return 0, False, f'There is no road from {n1} to {n2} in the dataset'
        
        if current_day > dataset['numDays']:
            warning_message = f'The itinerary is longer than the allowed number of days'
            break
        
        remaining_battery -= edge_length[(n1, n2)]
        if remaining_battery < 0:
            warning_message = f'Battery depleted on day {current_day} when traveling from {n1} to {n2}'
            break

        
        
        if (n1, n2) not in visited_edges:
            total_covered_length += edge_length[(n1, n2)]
            visited_edges.add((n1, n2))
            visited_edges.add((n2, n1))

    success_message = 'Solution is valid'
    score = total_covered_length
    if total_covered_length == max_possible_coverage:
        full_remaining_days = dataset['numDays'] - current_day
        battery_left = remaining_battery
        bonus = (full_remaining_days + (battery_left / dataset['batteryCapacity'])) / dataset['numDays'] + 1
        score = math.ceil(score * bonus)
        success_message = f'Solution is valid and finishes early! Multiplier bonus applied: x{bonus:.5f}'
        
    if warning_message:
        success_message += '. WARNING: ' + warning_message
    
    return score, True, success_message

if __name__ == '__main__':
    with open(SUBMISSION_FILE) as fi:
        solution = fi.read()
    with open(DATASET_FILE) as fi:
        dataset = fi.read()
    score, is_valid, message = getSolutionScore(solution, dataset)

    if is_valid:
        print('✅ Solution is valid!')
        print(f'Score: {score:_}')
    else:
        print('❌ Solution is invalid')

    print(f'Message: {message}')
    
