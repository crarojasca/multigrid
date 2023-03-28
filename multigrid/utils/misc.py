import functools



@functools.cache
def clip(value: int, low: int, high: int):
    return max(low, min(high, value))

@functools.cache
def get_view_exts(agent_dir: int, agent_pos: tuple[int, int], agent_view_size: int):
    """
    Get the extents of the square set of tiles visible to an agent.
    Note: the bottom extent indices are not included in the set.
    """
    # Facing right
    if agent_dir == 0:
        topX = agent_pos[0]
        topY = agent_pos[1] - agent_view_size // 2
    # Facing down
    elif agent_dir == 1:
        topX = agent_pos[0] - agent_view_size // 2
        topY = agent_pos[1]
    # Facing left
    elif agent_dir == 2:
        topX = agent_pos[0] - agent_view_size + 1
        topY = agent_pos[1] - agent_view_size // 2
    # Facing up
    elif agent_dir == 3:
        topX = agent_pos[0] - agent_view_size // 2
        topY = agent_pos[1] - agent_view_size + 1
    else:
        assert False, "invalid agent direction"

    botX = topX + agent_view_size
    botY = topY + agent_view_size

    return topX, topY, botX, botY
