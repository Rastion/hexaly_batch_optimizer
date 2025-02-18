import hexaly.optimizer
from qubots.base_optimizer import BaseOptimizer
from qubots.base_problem import BaseProblem

class HexalyBatchOptimizer(BaseOptimizer):
    """
    Hexaly-based optimizer for Batch Scheduling Problems
    Compatible with qubots' BatchSchedulingProblem.
    """
    
    def __init__(self, time_limit=30):
        self.time_limit = time_limit

    def optimize(self, problem: BaseProblem, initial_solution=None, **kwargs):
        # Extract problem parameters from the qubots problem instance
        model_params = {
            'nb_tasks': problem.nb_tasks,
            'nb_resources': problem.nb_resources,
            'capacity': problem.capacity,
            'types': problem.types,
            'resources': problem.resources,
            'duration': problem.duration,
            'nb_successors': problem.nb_successors,
            'successors': problem.successors,
            'nb_tasks_per_resource': problem.nb_tasks_per_resource,
            'time_horizon': problem.time_horizon
        }
        
        # Solve with Hexaly
        hexaly_solution = self._solve_with_hexaly(model_params)
        
        # Convert Hexaly solution to qubots format
        return self._format_solution(hexaly_solution, problem)

    def _solve_with_hexaly(self, params):
        """Core Hexaly optimization logic"""
        with hexaly.optimizer.HexalyOptimizer() as optimizer:
            model = optimizer.model

            # Batch content definition
            batch_content = [
                [model.set(params['nb_tasks_per_resource'][r]) 
                 for _ in range(params['nb_tasks_per_resource'][r])]
                for r in range(params['nb_resources'])
            ]

            # Constraints setup
            self._add_constraints(model, params, batch_content)

            # Objective definition
            makespan = self._create_objective(model, params, batch_content)
            model.minimize(makespan)
            model.close()

            # Solver configuration
            optimizer.param.time_limit = self.time_limit
            solution = optimizer.solve()
            
            return {
                'makespan': makespan.value,
                'batch_intervals': self._extract_intervals(solution, params)
            }

    def _add_constraints(self, model, params, batch_content):
        """Add problem-specific constraints to the model"""
        # Partition constraints
        for r in range(params['nb_resources']):
            model.constraint(model.partition(model.array(batch_content[r])))

        # Type and capacity constraints
        for r in range(params['nb_resources']):
            resource_types = model.array(params['types_in_resource'][r])
            type_lambda = model.lambda_function(lambda i: resource_types[i])
            
            for batch in batch_content[r]:
                model.constraint(model.count(model.distinct(batch, type_lambda)) <= 1)
                model.constraint(model.count(batch) <= params['capacity'][r])

    def _create_objective(self, model, params, batch_content):
        """Create makespan objective"""
        batch_intervals = [
            [model.interval(0, params['time_horizon']) for _ in batches]
            for batches in batch_content
        ]
        
        # Precedence constraints
        task_intervals = [None] * params['nb_tasks']
        for t in range(params['nb_tasks']):
            r = params['resources'][t]
            batch_idx = model.find(model.array(batch_content[r]), t)
            task_intervals[t] = model.at(model.array(batch_intervals[r]), batch_idx)
            model.constraint(model.length(task_intervals[t]) == params['duration'][t])

        for t in range(params['nb_tasks']):
            for s in params['successors'][t]:
                model.constraint(task_intervals[t] < task_intervals[s])

        return model.max([model.end(iv) for resource in batch_intervals for iv in resource])

    def _extract_intervals(self, solution, params):
        """Extract interval data from Hexaly solution"""
        return {
            r: [
                (solution.start(iv), solution.end(iv))
                for iv in solution.model.batch_intervals[r]
            ]
            for r in range(params['nb_resources'])
        }

    def _format_solution(self, hexaly_solution, problem):
        """Convert Hexaly solution to qubots batch_schedule format"""
        batch_schedule = []
        
        for r in range(problem.nb_resources):
            for batch_idx, (start, end) in enumerate(hexaly_solution['batch_intervals'][r]):
                tasks = [
                    t for t in range(problem.nb_tasks)
                    if problem.resources[t] == r
                    and start <= hexaly_solution['task_intervals'][t].start() < end
                ]
                
                batch_schedule.append({
                    'resource': r,
                    'tasks': tasks,
                    'start': start,
                    'end': end
                })
        
        return {'batch_schedule': batch_schedule}, hexaly_solution['makespan']