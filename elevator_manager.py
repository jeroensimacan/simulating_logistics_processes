
import numpy as np
import pandas as pd
import simpy

from elevator import Elevator


class ElevatorManager:
    class MoveGoal:
        """
            wraps users so we can track where they are going to
        """

        def __init__(self, user, start_floor, dest_floor, onboard_event):
            self.user = user
            self.start_floor = start_floor
            self.dest_floor = dest_floor
            self.onboard_event = onboard_event
            self.arrive_event = None
            self.elevator = None  # used elevator is unknown until loaded!

    def __init__(self, env, settings, name):
        self.env = env
        self.name = name
        self.retrigger = self.env.event()

        # queue for passengers waiting to get on elevator, one queue per floor
        self.boarding_queues = {f: [] for f in settings['floors']}
        # self.boarding_queues = {f: [] for f in range(1, floors + 1)}

        # events to notify passengers when they have arrived at their floor, one per floor
        self.arrive_events = {el: {f: simpy.Event(env) for f in settings['floors']} for el in settings['elevators']}
        self.elevators = []
        for key, elevator_settings in settings['elevators'].items():
            el = Elevator(env, elevator_settings, settings['floors'], self.boarding_queues, self.arrive_events,
                          self.retrigger, key)
            self.elevators.append(el)
        self.history = []

        self.distribution = simpy.Resource(env, capacity=100_000)
        self.elevator = simpy.Resource(env, capacity=100_000)
        self.env.process(self.retrigger_function())

    def retrigger_function(self):
        while True:
            yield self.retrigger
            print(f'{self.env.now:.2f} Retriggered, hence we should ask which elevators are available.')
            b = 0
            from_floor = self.retrigger.value['from_floor']
            to_floor = self.retrigger.value['to_floor']
            branch = self.boarding_queues[from_floor][0].user.branch
            self.history.append([self.env.now, from_floor, to_floor, 0, 0, branch, True])
            self.trigger_elevator(**self.retrigger.value)
            self.retrigger = self.env.event()
            for elevator in self.elevators:
                elevator.retrigger = self.retrigger

    def trigger_elevator(self, from_floor, to_floor):
        # implement smart retriggering of elevators!
        elevator_status = [el.status for el in self.elevators]
        elevator_floors = [el.on_floor for el in self.elevators]
        # elevator_direction = [el.move_inc for el in self.elevators]
        # elevator_full = [el.passenger_cap == el.current_load for el in self.elevators]

        if all([el == 'ACTIVE' for el in elevator_status]):
            # no need to trigger any elevators!
            pass

        elif all([el == 'IDLE' for el in elevator_status]):
            # pick closest one!
            floor_diff = np.abs(np.array(elevator_floors) - from_floor)
            idx = np.random.choice(np.argwhere(floor_diff == np.min(floor_diff)).flatten())
            self.elevators[idx].reactivate.succeed()
            self.elevators[idx].reactivate = self.env.event()
        else:
            # check whether the active elevators are moving in the direction of the client
            suitable_elevators = []
            for elevator in self.elevators:
                if elevator.status == 'IDLE':
                    continue
                elif elevator.on_floor == from_floor and not elevator.can_load():
                    continue

                if elevator.on_floor == from_floor:
                    suitable_elevators.append(elevator)
                elif elevator.on_floor + elevator.move_inc == from_floor:
                    suitable_elevators.append(elevator)
                elif from_floor - elevator.on_floor > from_floor - (elevator.on_floor + elevator.move_inc):
                    suitable_elevators.append(elevator)
            if len(suitable_elevators) == 0:
                idx = np.where([el == 'IDLE' for el in elevator_status])[0][0]
                self.elevators[idx].reactivate.succeed()
                self.elevators[idx].reactivate = self.env.event()
            else:
                # at least one of the elevators is moving in the right direction, hence we do not need to trigger..
                pass

    def move_to(self, user, from_floor, to_floor):
        """
            Return an event that fires when the user gets on the elevator
            The event returns another event that fires when the passager
            arrives at their destination floor

            (uses the env.process() to convert a process to a event)

        """

        return self.env.process(self._move_to(user, from_floor, to_floor))

    def _move_to(self, user, from_floor, to_floor):

        """
            Puts the user into a queue for the elevator
        """

        # creat event to notify user when they can get onto the elemator
        onboard_event = simpy.Event(self.env)

        # save move data in a wrapper and put user into queue
        move_goal = self.MoveGoal(user, from_floor, to_floor, onboard_event)
        if len(self.boarding_queues[from_floor]) == 0:
            triggered = True
        else:
            triggered = False
        self.boarding_queues[from_floor].append(move_goal)

        self.trigger_elevator(from_floor, to_floor)

        user.add_event(time=self.env.now, event='waiting', process='elevator', location=self.name)

        if user.area == 0.35:
            nr_passengers, nr_load_carriers = 0, 1
        else:
            nr_passengers, nr_load_carriers = 1, 0
        self.history.append(
            [self.env.now, from_floor, to_floor, nr_passengers, nr_load_carriers, user.branch, triggered])

        # wait for elevator to arrive, and have space for user
        print(f'{self.env.now:.2f}: Yield of onboard event of {user.user_type}: {user.id}')
        yield onboard_event
        print(f'{self.env.now:.2f}: Arrival of {user.user_type}: {user.id}')
        self.history.append(
            [self.env.now, from_floor, to_floor, nr_passengers * -1, nr_load_carriers * -1, user.branch, False])

        # get destination arrival event
        # dest_event = self.arrive_events[move_goal.elevator][to_floor]
        # move_goal.arrive_event = dest_event
        # print(f'{self.env.now:.2f}: Arrival of load carrier: {user.id}')
        # return dest_event

    def to_pandas(self):
        df = pd.DataFrame(self.history,
                          columns=['_time', 'from_floor', 'to_floor', 'personnel diff', 'lc diff', 'branch',
                                   'triggered'])
        df['queue length personnel'] = np.nan
        df['queue length lc'] = np.nan
        for key in self.boarding_queues.keys():
            df['queue length personnel'].update(df.loc[df['from_floor'] == key, 'personnel diff'].cumsum())
            df['queue length lc'].update(df.loc[df['from_floor'] == key, 'lc diff'].cumsum())

        df['hour'] = df['_time'] // 3600
        df['minute'] = df['_time'] % 3600 // 60
        df['name'] = self.name
        df.reset_index(drop=True, inplace=True)
        return df
