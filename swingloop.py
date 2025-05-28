# Swing loop will swing back and forth through iterarable until a certain condition is met

class CycleLoop:
    '''
    Creates an object that iterate through list, and cycle back round once the current index is 0 or len(list)
    '''

    def __init__(self, iterable:list, start_index:int):

        if len(iterable) == 0:
            raise RuntimeError("Cyceloop() was given list with no items")

        self.iterable = iterable
        self.START_INDEX = start_index
        self.MAX_INDEX = len(iterable) - 1

        self.current = iterable[start_index]
        self.current_index = self.START_INDEX


    def step(self, stp:int):
        '''
        steps through list by stp indexes
        '''
        self.current_index += stp

        if self.current_index > self.MAX_INDEX:
            self.current_index = 0

        elif self.current_index < 0:
            self.current_index = self.MAX_INDEX

        self.current = self.iterable[self.current_index]


    def reset(self):
        '''
        Resets the current index to start index
        '''
        self.current_index = self.START_INDEX

    
    def peek1(self, stp: int) -> int|float :
        '''
        Returns self[current_index + 1] without changing the state of the Cycleloop
        '''
        assert stp == 1 or stp == -1

        return self.iterable[self.current_index + stp ]

            


sl = CycleLoop(["steve","frank","bob","dave"], 2)

print(sl.current)
print(sl.peek1(1))
print(sl.peek1(-1))

