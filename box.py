'''Provides a bounding box class.'''


class BoundingBox(object):
    """Represents a BoundingBox."""

    def __init__(self, bb_x, bb_y, bb_w, bb_h):
        self.bb_x = bb_x
        self.bb_y = bb_y
        self.bb_w = bb_w
        self.bb_h = bb_h

    @property
    def center(self):
        """Bounding box center."""
        return (int(self.bb_x + self.bb_w / 2), int(self.bb_y + self.bb_h / 2))

    @property
    def top_left(self):
        """Bounding top left."""
        return (self.bb_x, self.bb_y)

    @property
    def bottom_right(self):
        """Bounding bottom right."""
        return (self.bb_x + self.bb_w, self.bb_y + self.bb_h)

    def intersects(self, box):
        """Return whether a given box intersects with this one."""
        return not (
            # box is to the left or right
            box.bb_x > self.bb_x + self.bb_w or self.bb_x > box.bb_x + box.bb_w
            or
            # box is above or below
            box.bb_y > self.bb_y + self.bb_h
            or self.bb_y > box.bb_y + box.bb_h)

    def __contains__(self, center):
        """Return whether a given box intersects with this one."""
        return center[0] > self.bb_x and center[0] < self.bb_x + self.bb_w
