
from dataclasses import dataclass
from typing import List

from load_carrier import User


@dataclass
class Truck:

    eta: float
    load_carriers: List[User]
    business_partner: str
    # supplier: str
    capacity: int = 50
    load: int = 0
    departure_time: float = -1.

    def __init__(self, eta, business_partner, capacity=50):
        self.eta = eta
        self.business_partner = business_partner
        self.load_carriers = []

    def add_load_carrier(self, load_carrier):
        self.load_carriers.append(load_carrier)
        self.load += 1

    def remove_load_carrier(self, index):
        if self.load > 0:
            load_carrier = self.load_carriers.pop(index)
            self.load -= 1
            return load_carrier
        else:
            return None

    def set_departure_time(self, time):
        self.departure_time = time
