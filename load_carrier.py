
import pandas as pd


class User:
    """
        simple class with unique id per user
    """

    next_id: int = 1

    @classmethod
    def get_next_id(cls):
        id = cls.next_id
        cls.next_id += 1

        return id

    def __init__(self, user_type, current_floor=1, destination_floor=2, branch='', area=0.5, elevator=None, x=None, y=None, z=None):
        self.id = self.get_next_id()
        self.current_floor = current_floor
        self.destination_floor = destination_floor
        self.history = pd.DataFrame(columns=['time', 'event', 'process', 'location'])
        self.branch = branch
        self.area = area
        self.elevator = elevator
        self.x = x
        self.y = y
        self.z = z
        self.user_type = user_type

    def add_event(self, time, process, event, location):
        row = pd.DataFrame(data={'time': [time], 'event': [event], 'process': [process], 'location': [location]})
        self.history = self.history.append(row)
        self.history.reset_index(drop=True, inplace=True)

    def get_history(self):
        history = self.history
        history['user_type'] = self.user_type
        history['branch'] = self.branch
        history['id'] = self.id
        history['duration'] = history['time'].shift(-1) - history['time']
        return history
