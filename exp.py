# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.


import random
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    def __init__(self, env, env_config, resc):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, env_config)
        self.walk_stack = Stack()  # a stack to store the movements
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is a seq number of the victim,(x,y) the position, <vs> the list of vital signals
        
        #cria um conjunto das posições que visitou
        self.visited = set()
        #já visitou a base
        self.visited.add((0,0))

        self.dfs_stack = []

        self.priority = {
        "EXP_2": [0,1,2,3,4,5,6,7],
        "EXP_3": [2,3,4,5,6,7,0,1],
        "EXP_1": [6,7,0,1,2,3,4,5]
        }.get(
        self.NAME,
        list(range(8))
        )

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        
        # Cost to move diagonally, read vital signs, and return,
        # assuming the maximum difficulty is 3.    
        self.one_more_step = self.COST_DIAG*2*3 + self.COST_READ

    def get_next_position(self):
        """ Randomically, gets the next position that can be explored (no wall and inside the grid)
            There must be at least one CLEAR position in the neighborhood, otherwise it loops forever.
        """
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()

        # Loop until a CLEAR position is found
        #while True:
            # Get a random direction
            #direction = random.randint(0, 7)
            # Check if the corresponding position in walls_and_lim is CLEAR
            #if obstacles[direction] == VS.CLEAR:
            #    return Explorer.AC_INCR[direction]

        #Trocando pela parte da DFS

        for direction in self.priority:

            #se a direção não está livre, ignoro
            if obstacles[direction] != VS.CLEAR:
                continue
            
            #vê quanto anda pra chegar lá
            dx, dy = Explorer.AC_INCR[direction]

            #calcula nova pos
            nx = self.x + dx
            ny = self.y + dy

            #se ainda não visitou essa posição pra não ficar repentindo
            if(nx, ny) not in self.visited:
                return dx, dy

        return None
        
    def explore(self):
        # busca posição disponível para avançar      
        next_pos = self.get_next_position()

        # SEM VIZINHOS LIVRES, FAZ BACKTRACKING
        if next_pos is None:

            # pilha tá vazia, exploração acabou
            if not self.dfs_stack:
                return

            # vê pra onde vai voltar
            px, py = self.dfs_stack.pop()

            # vê quanto se mexe pra voltar pra lá
            dx = px - self.x
            dy = py - self.y

            # anda de volta
            result = self.walk(dx, dy)

            # se deu boa
            if result == VS.EXECUTED:
                self.walk_stack.push((dx, dy))
                self.x = px
                self.y = py

            return

        # AVANÇA PRA UMA POSIÇÃO NÃO VISITADA
        dx, dy = next_pos

        # empilha a pos atual pra saber pra onde voltar
        self.dfs_stack.append((self.x, self.y))

        # Mede a bateria antes do movimento
        rtime_bef = self.get_rtime()   ## get remaining batt time before the move
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()   ## get remaining batt time after the move

        # Test the result of the walk action
        # It should never bump, since get_next_position always returns a valid position...
        # but for safety, let's test it anyway
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        # se deu boa
        if result == VS.EXECUTED:
            # puts the visited position in a stack. When the batt is low, 
            # the explorer unstack each visited position to come back to the base
            self.walk_stack.push((dx, dy))

            # update the agent's position relative to the origin of 
            # the coordinate system used by the agents
            self.x += dx
            self.y += dy   

            self.visited.add((self.x,self.y))       

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[seq] = ((self.x, self.y), vs)
                #print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):
        """ Procedure to return to the base: pops the walk_stack to follow
        the exploration path in the opposite direction """
  
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        # Walk resulted in bumping into a wall or end of grid
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
            
        # Walk succeded
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        
    def deliberate(self) -> bool:
        """  The simulator calls this method at each cycle. 
        Must be implemented in every agent. The agent chooses the next action.
        """

        consumed_time = self.TLIM - self.get_rtime()
        
        # check if it is time to come back to the base      
        if (consumed_time + self.one_more_step) < self.get_rtime():
            # continue to explore
            self.explore()
            return True

        # Returning to the base terminates when there are no more moves to pop from the stack
        if self.walk_stack.is_empty():
            # time to wake up the rescuer
            # pass the walls and the victims (here, they're empty)
            print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the MASTER rescuer")
            self.resc.merge_maps(self.NAME, self.map, self.victims)
            return False

        # move to the base
        self.come_back()
        return True



