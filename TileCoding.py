'''
Created on Jun 3, 2016

@author: mgomrokchi
'''

import random 
import math
import operator


_maxnumfloats = 20  # maximum number of variables used in one grid
_maxLongint = 2147483647  # maximum integer
_maxLongintBy4 = int(_maxLongint / 4)  # maximum integer divided by 4   
_randomTable = [random.randrange(_maxLongintBy4) for i in range(2048)]  # table of random numbers

# The following are temporary variables used by tiles.
_qstate = [0 for i in range(_maxnumfloats)]
_base = [0 for i in range(_maxnumfloats)]
_safteydict = {'unsafe':0, 'safe':1, 'super safe':2}
_UNSAFE = 0
_SAFE = 1
_SUPER_SAFE = 2

class CollisionTable:
    "Structure to handle collisions"
    def __init__(self, sizeval=2048, safetyval='safe'):
        # if not power of 2 error
        if not powerOf2(sizeval):
            print("error - size should be a power of 2")
        self.size = sizeval                        
        self.safety = _safteydict[safetyval]            # one of 'safe', 'super safe' or 'unsafe'
        self.calls = 0
        self.clearhits = 0
        self.collisions = 0
        self.data = [-1 for i in range(self.size)]  
    def __str__(self):
        #print("Prepares a string for printing whenever this object is printed")
        return "Collision table: " + \
               " Safety : " + str(self.safety) + \
               " Usage : " + str(self.usage()) + \
               " Size :" + str(self.size) + \
               " Calls : "+ str(self.calls) + \
               " Collisions : " + str(self.collisions)
    
    def print_ (self):
        #"Prints info about collision table"
        print ("usage: " + str(self.usage())+ " size: "+ str(self.size)+ " calls: "+str(self.calls)+ " clearhits: " + str(self.clearhits)+" collisions: "+ str(self.collisions)+ " safety: "+ str(self.safety))

    def reset (self):
        #"Reset Ctable values"
        self.calls = 0
        self.clearhits = 0
        self.collisions = 0
        self.data = [-1 for i in range(self.size)]
    
    def stats (self):
        #"Return some statistics of the usage of the collision table"
        return self.calls, self.clearhits, self.collisions, self.usage

    def usage (self):
        #"count how many entries in the collision table are used"
        use = 0
        for d in self.data:
            if d >= 0:
                use += 1
        return use

def startTiles (coordinates, numtilings, floats, ints=[]):
    #"Does initial assignments to _coordinates, _base and _qstate for both GetTiles and LoadTiles"
    global _base, _qstate
    numfloats = len(floats)
    i = numfloats + 1                   # starting place for integers
    for v in ints:                      # for each integer variable, store it
        coordinates[i] = v             
        i += 1
    i = 0
    for float in floats:                # for real variables, quantize state to integers
        _base[i] = 0
        _qstate[i] = int(math.floor(float * numtilings))
        i += 1
    
def fixcoord (coordinates, numtilings, numfloats, j):
    #"Fiddles with _coordinates and _base - done once for each tiling"
    global _base, _qstate
    for i in range(numfloats):          # for each real variable
        if _qstate[i] >= _base[i]:
            coordinates[i] = _qstate[i] - ((_qstate[i] - _base[i]) % numtilings)
        else:
            coordinates[i] = _qstate[i]+1 + ((_base[i] - _qstate[i] - 1) % numtilings) - numtilings
        _base[i] += 1 + (2*i)
        #_hashnum[i] = _randomTable[(coordinates[i] + _increment[i]) & 2047]
    coordinates[numfloats] = j

def hashUNH (ints, numInts, m, increment=449):
    "Hashing of array of integers into below m, using random table"
    res = 0
    for i in range(numInts):
        res += _randomTable[(ints[i] + i*increment) % 2048]
        #res += _randomTable[(ints[i] + i*increment) & 2047] 

    #res = reduce(operator.add, [_randomTable[(ints[i] + i*increment) & 2047] for i in xrange(numInts)])
    #res = reduce(operator.add,_hashnum)
    return res % m

def hash (ints, numInts, ct):
    "Returns index in collision table corresponding to first part of ints (an array)"
    ct.calls += 1
    memSize = ct.size
    j = hashUNH(ints, numInts, memSize)
    if ct.safety == _SUPER_SAFE:
        ccheck = ints[:]                # use whole list as check
    else:                               # for safe or unsafe, use extra hash number as check
        ccheck = hashUNH(ints, numInts, _maxLongint, 457)
    if ccheck == ct.data[j]:            # if new data same as saved data, add to hits
        ct.clearhits += 1
    elif ct.data[j] < 0:                # first time, set up data
        ct.clearhits += 1
        ct.data[j] = ccheck
    elif ct.safety == _UNSAFE:         # collison, but we don't care   
        ct.collisions += 1
    else:                               # handle collision - rehash
        h2 = 1 + 2*hashUNH(ints, numInts, _maxLongintBy4)
        i = 1
        while ccheck != ct.data[j]:     # keep looking for a new spot until we find an empty spot
            ct.collisions += 1
            j = (j + h2) % memSize
            if i > memSize:             # or we run out of space 
                print("Tiles: Collision table out of memory")
                return -1               # force it to stop if out of memory
            if ct.data[j] < 0:          
                ct.data[j] = ccheck
            i += 1
    return j

def powerOf2 (n):
    lgn = math.log(n, 2)
    return (lgn - math.floor(lgn)) == 0


def mod(num, by):
    if num >= 0:
        return num % by
    else:
        return (by + (num % by)) % by
    
def fixcoordwrap(coordinates, numtilings, numfloats, j, wrapwidths):
    global _widthxnumtilings, _qstate, _base
    for i in range(numfloats):  # loop over each relevant dimension
        # find coordinates of activated tile in tiling space 
        #if _qstate[i] >= _base[i]:
        coordinates[i] = _qstate[i] - ((_qstate[i] - _base[i]) % numtilings)
        #else:
        #    _coordinates[i] = _qstate[i]+1 + ((_base[i] - _qstate[i] - 1) % numtilings) - numtilings
        if wrapwidths[i] != 0:
            #_coordinates[i] = mod(_coordinates[i], _widthxnumtilings[i])
            coordinates[i] = coordinates[i] % _widthxnumtilings[i]
        _base[i] += 1 + (2 * i) # compute displacement of next tiling in quantized space
    coordinates[numfloats] = j # add indices for tiling and hashing_set so they hash differently


def tiles (numtilings, memctable, floats, ints=[]):
    '''Returns list of numtilings tiles corresponding to variables (floats and ints),
        hashed down to mem, using ctable to check for collisions'''

    if isinstance(memctable, CollisionTable):
        hashfun = hash
    else:
        hashfun = hashUNH

    numfloats = len(floats)
    numcoord = 1 + numfloats + len(ints)
    _coordinates = [0]*numcoord
    startTiles (_coordinates, numtilings, floats, ints)
    tlist = [None] *numtilings
    for j in range(numtilings):             # for each tiling
        fixcoord(_coordinates, numtilings, numfloats, j)
        hnum = hashfun(_coordinates, numcoord, memctable)
        tlist[j] = hnum
    return tlist

def loadtiles (tiles, startelement, numtilings, memctable, floats, ints=[]):
    '''Loads numtilings tiles into array tiles, starting at startelement, corresponding
       to variables (floats and ints), hashed down to mem, using ctable to check for collisions'''

    if isinstance(memctable, CollisionTable):
        hashfun = hash
    else:
        hashfun = hashUNH

    numfloats = len(floats)
    numcoord = 1 + numfloats + len(ints)
    _coordinates = [0]*numcoord
    startTiles (_coordinates, numtilings, floats, ints)
    for j in range(numtilings):
        fixcoord(_coordinates, numtilings, numfloats, j)
        hnum = hashfun(_coordinates, numcoord, memctable)
        tiles[startelement + j] = hnum
        
def tileswrap(numtilings, memctable, floats, wrapwidths, ints=[]):
    '''Returns list of numtilings tiles corresponding to variables (floats and ints),
        hashed down to mem, using ctable to check for collisions - wrap version'''
    global _widthxnumtilings

    if isinstance(memctable, CollisionTable):
        hashfun = hash
    else:
        hashfun = hashUNH

    numfloats = len(floats)
    numcoord = 1 + numfloats + len(ints)
    _coordinates = [0]*numcoord
    tiles = [None] * numtilings
    startTiles (_coordinates, numtilings, floats, ints)
    _widthxnumtilings = [wrapwidths[i] * numtilings for i in range(numfloats)]
    for j in  range(numtilings):
        fixcoordwrap(_coordinates, numtilings, numfloats, j, wrapwidths)
        hnum = hashfun(_coordinates, numcoord, memctable)
        tiles[j] = hnum
    return tiles

def loadtileswrap(tiles, startelement, numtilings, memctable, floats, wrapwidths, ints=[]):
    '''Returns list of numtilings tiles corresponding to variables (floats and ints),
        hashed down to mem, using ctable to check for collisions - wrap version'''
    global _widthxnumtilings

    if isinstance(memctable, CollisionTable):
        hashfun = hash
    else:
        hashfun = hashUNH

    numfloats = len(floats)
    numcoord = 1 + numfloats + len(ints)
    _coordinates = [0]*numcoord
    startTiles (_coordinates, numtilings, floats, ints)
    _widthxnumtilings = [wrapwidths[i] * numtilings for i in range(numfloats)]
    for j in  range(numtilings):
        fixcoordwrap(_coordinates, numtilings, numfloats, j, wrapwidths)
        hnum = hashfun(_coordinates, numcoord, memctable)
        tiles[startelement + j] = hnum

    
    #def stateAggegatorApproximator(self):
    #    return phi
    
    #def polyApproximator(self):
        #return Phi
    
    #def gaussianApproximator(self):
        #return Phi
    
    
