
import datetime

import pandas as pd
import random
import simpy
import numpy as np
from scipy.stats import uniform


class Elevator:
    """
        Elevator that move people from floor to floor
        Has a max compatity
        Uses a event to notifiy passengers when they can get on the elevator
        and when they arrive at their destination floor
    """

    next_id = 1

    @classmethod
    def get_next_id(cls):
        id = cls.next_id
        cls.next_id += 1

        return id

    def __init__(self, env, settings, floors, boarding_queues, arrive_events, retrigger, name):
        self.id = self.get_next_id()
        self.name = name

        self.env = env
        self.floors = floors
        self.on_floor = self.floors[0]
        self.move_inc = 1
        self.current_load = 0
        self.area = settings['length'] * settings['width']

        self.boarding_queues = boarding_queues
        self.arrive_events = arrive_events

        # list of passengers on elevator, one per floor
        self.on_board = {f: [] for f in self.floors}

        # start elevator
        self.moving = env.process(self._move_next_floor())
        self.history = []
        self.reactivate = self.env.event()
        self.status = 'IDLE'
        self.retrigger = retrigger

        # insert all distributions to draw from:
        self.unloading_time = uniform(settings['unloading_time']['min'], settings['unloading_time']['max'])
        self.loading_time = uniform(settings['loading_time']['min'], settings['loading_time']['max'])
        self.closing_time = settings['door_close_time']
        self.move_time = settings['move_time']

        self.repairman = simpy.Resource(self.env, capacity=1)
        self.env.process(self.break_elevator())

        self.broken = False
        self.door_status = 'CLOSED'
        self.outage_probability = settings['failure']['probability']
        self.outage_duration = settings['failure']['duration']

    def time_to_failure(self):
        return np.random.uniform(low=0.5 * 1 / self.outage_probability, high=1.5 * 1 / self.outage_probability)

    def break_elevator(self):
        while True:
            yield self.env.timeout(self.time_to_failure())
            if not self.broken:
                self.moving.interrupt()

    def calculate_load(self):
        return sum([el.user.area for v in self.on_board.values() for el in v])

    def can_load(self):
        return self.calculate_load() <= self.area * 0.8

    def _move_next_floor(self):
        """
            Moves the elevator up and down
            Elevator stops at every floor
        """

        def _update_direction():
            update_time = self.move_time
            floors_below = [f for f in self.floors if f < self.on_floor]
            floors_above = [f for f in self.floors if f > self.on_floor]

            customers_below = [len(self.on_board[floor]) + len(self.boarding_queues[floor]) for floor in floors_below]
            customers_above = [len(self.on_board[floor]) + len(self.boarding_queues[floor]) for floor in floors_above]
            if sum(customers_below) == 0 and sum(customers_above) == 0:
                update_time = 0
            else:
                if (sum(customers_below) == 0 and self.move_inc == -1) or \
                        (sum(customers_above) == 0 and self.move_inc == 1):
                    print(f'Elevator {self.id}: Smart change of direction!')
                    self.move_inc *= -1
                self.history.append([self.env.now, self.calculate_load(), self.on_floor, 'START MOVING'])

                idx_next_floor = self.floors.index(self.on_floor) + self.move_inc
                next_floor = self.floors[idx_next_floor]
                diff_floor = next_floor - self.on_floor
                self.on_floor = next_floor
                update_time *= abs(diff_floor)
            return update_time

        def _determine_unloading_time():
            # unloading time of an rc == 10
            # unloading time of personnel == 2
            unloading_time = 0
            for unboarder in self.on_board[self.on_floor]:
                if unboarder.user.area == 0.35:
                    # this is a person
                    unloading_time += 2
                else:
                    # this is a load carrier
                    unloading_time += 10
            return unloading_time

        def _unload_arriving_passengers():
            while len(self.on_board[self.on_floor]) > 0:
                p = self.on_board[self.on_floor].pop()
                p.user.add_event(time=self.env.now, event='unloading', process='elevator', location=self.name)
                p.user.current_floor = self.on_floor
                p.onboard_event.succeed()

            arrive_events = self.arrive_events[self.name][self.on_floor]
            self.arrive_events[self.name][self.on_floor] = simpy.Event(self.env)
            arrive_events.succeed()

        def _load_departing_passengers():
            boarding = []
            current_load = self.calculate_load()
            for el in self.boarding_queues[self.on_floor]:
                loaded_users = [el for v in self.on_board.values() for el in v]
                if len(loaded_users) > 0:
                    user_type = loaded_users[0].user.user_type
                    if el.user.user_type != user_type:
                        continue
                if current_load + el.user.area < 0.8 * self.area:
                    boarding.append(el)
                    current_load += el.user.area

            for b in boarding:
                self.boarding_queues[self.on_floor].remove(b)
                b.arrive_event = self.arrive_events[self.name][b.dest_floor]
                b.elevator = self.name
                b.user.add_event(time=self.env.now, event='loading', process='elevator', location=self.name)
                self.on_board[b.dest_floor].append(b)

            return np.sum(self.loading_time.rvs(len(boarding)))

        def _has_task():
            if sum([len(v) for v in self.boarding_queues.values()]) > 0:
                return True
            elif sum([len(v) for v in self.on_board.values()]) > 0:
                return True
            return False

        while True:
            try:
                if not _has_task():
                    self.status = 'IDLE'
                    self.history.append([self.env.now, self.calculate_load(), self.on_floor, 'IDLE'])
                    yield self.reactivate
                    print(f'{self.env.now:.2f} Triggered reactivation of elevator {self.id}')
                    self.status = 'ACTIVE'
                else:
                    unloading_time = _determine_unloading_time()
                    if unloading_time > 0:
                        yield self.env.process(self.open_door())

                    self.history.append([self.env.now, self.calculate_load(), self.on_floor, 'START UNLOADING'])
                    yield self.env.timeout(unloading_time)
                    _unload_arriving_passengers()

                    while True:
                        loading_time = _load_departing_passengers()
                        if loading_time > 0:
                            yield self.env.process(self.open_door())
                        else:
                            break
                        self.history.append([self.env.now, self.calculate_load(), self.on_floor, 'START LOADING'])
                        yield self.env.timeout(loading_time)

                    yield self.env.process(self.close_door())

                    if len(self.boarding_queues[self.on_floor]) > 0:
                        self.retrigger.succeed(value={
                            'from_floor': self.boarding_queues[self.on_floor][0].start_floor,
                            'to_floor': self.boarding_queues[self.on_floor][0].dest_floor
                        })

                    move_time = _update_direction()
                    yield self.env.timeout(move_time)
            except simpy.Interrupt:
                self.broken = True
                self.history.append([self.env.now, self.calculate_load(), self.on_floor, 'BREAKDOWN'])
                with self.repairman.request() as request:
                    yield request
                    yield self.env.timeout(self.outage_duration)
                self.broken = False

    def open_door(self):
        if self.door_status != 'OPEN':
            self.history.append([self.env.now, self.calculate_load(), self.on_floor, 'OPEN DOORS'])
            self.door_status = 'OPEN'
            yield self.env.timeout(self.closing_time)

    def close_door(self):
        if self.door_status != 'CLOSED':
            self.history.append([self.env.now, self.calculate_load(), self.on_floor, 'CLOSING DOORS'])
            self.door_status = 'CLOSED'
            yield self.env.timeout(self.closing_time)

    def to_pandas(self):
        df = pd.DataFrame(self.history, columns=['_time', 'load', 'floor', 'reportingStatus'])
        # insert a task in between each section
        df['duration'] = df['_time'].shift(-1) - df['_time']
        df['aasCode'] = self.name
        df['reportingStatus'] = df['reportingStatus'].str.replace('START ', '')
        df.drop(index=df.loc[df['duration'] == 0].index, inplace=True)
        df['group'] = (df['reportingStatus'] != df['reportingStatus'].shift()).cumsum().rename('group')
        agg_df = df.groupby(by='group').agg({
            '_time': 'first',
            'load': 'first',
            'floor': 'first',
            'reportingStatus': 'first',
            'duration': 'sum',
            'aasCode': 'first'
        })
        return agg_df

