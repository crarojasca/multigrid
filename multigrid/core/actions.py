import enum



class Action(enum.IntEnum):
    """
    Enumeration of possible actions.
    """
    left = 0 #: Turn left
    right = 1 #: Turn right
    forward = 2 #: Move forward
    stay = 3 #: Stay in place
    pickup = enum.auto() #: Pick up an object
    drop = enum.auto() #: Drop an object
    toggle = enum.auto() #: Toggle / activate an object
    done = enum.auto() #: Done completing task
