from enum import Enum, auto


class Category(Enum):
    YATZY = 0  # Bit 0
    CHANCE = 1  # Bit 1
    FULL_HOUSE = 2  # Bit 2
    LARGE_STRAIGHT = 3  # Bit 3
    SMALL_STRAIGHT = 4  # Bit 4
    FOUR_OF_A_KIND = 5  # Bit 5
    THREE_OF_A_KIND = 6  # Bit 6
    TWO_PAIRS = 7  # Bit 7
    ONE_PAIR = 8  # Bit 8
    SIXES = 9  # Bit 9
    FIVES = 10  # Bit 10
    FOURS = 11  # Bit 11
    THREES = 12  # Bit 12
    TWOS = 13  # Bit 13
    ONES = 14  # Bit 14