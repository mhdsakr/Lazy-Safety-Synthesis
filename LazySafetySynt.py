#!/usr/bin/env python2.7
# coding=utf-8

"""
A basic implementation for safety synthesis that uses transition functions
instead of a complete transition relation. It also optimizes the two faunctions
walk and extract_output_funcs by using a hashtable and the restrict function.

----------------------
"""

status = ["TODO: check the case when output functions depend on error fake latch"]


import argparse
import logging
import sys
sys.path.insert(0, './pycudd')
import pycudd
from aiger_swig.aiger_wrap import *
from aiger_swig.aiger_wrap import aiger
import os
import time
import copy

transition_funcs = dict()

#don't change status numbers since they are used by the performance script
EXIT_STATUS_REALIZABLE = 10
EXIT_STATUS_UNREALIZABLE = 20

#Ddarray to be used in vector compose function
substitutionDdArray = None

#hash table for the walk method
bddToAigCache = dict()

#hash table for the bddtovalue method
andgateTobdd = dict()

#the 2 below variables to be used in converting aag
#max aiger var before generating output functions
firstMaxVar = 0
cont_input_lits = []
first_ands_num = 0


#for mixed model checking
accumErrorBDD = None

#: :type: aiger
spec = None

#: :type: DdManager
cudd = None

shortestPath = 0  # Resillient Guarantee

# error output can be latched or unlatched, in this case we introduce a latch for the error output
#: :type: aiger_symbol
error_fake_latch = None
errorVarIndex = -1

#: :type: Logger
logger = None


def ABCminimization(filePath):
    filePath = os.path.splitext(filePath)[0]
    #https://bitbucket.org/alanmi/abc/downloads
    abcCommand = """abc -c \"read {0}.aig; strash; refactor -zl; rewrite -zl;
     strash; refactor -zl; rewrite -zl; strash; refactor -zl; rewrite -zl;
      dfraig; rewrite -zl; dfraig; write {1}.aig\" ./run > /dev/null"""
    #AIGER Toolset (http://fmv.jku.at/aiger/aiger-1.9.4.tar.gz)
    aagToaig = 'aigtoaig {0}.aag {1}.aig'
    aigToaag = 'aigtoaig {0}.aig {1}.aag'
    os.system(aagToaig.format(filePath, filePath))
    os.system(abcCommand.format(filePath, filePath))
    os.system(aigToaag.format(filePath, filePath))
    os.remove(filePath + '.aig')


def setup_logging(verbose):
    global logger
    level = None
    if verbose is 0:
        level = logging.INFO
    elif verbose >= 1:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)-10s%(message)s",
                        datefmt="%H:%M:%S",
                        level=level,
                        stream=sys.stdout)

    logger = logging.getLogger(__name__)

#check if lit is odd
def is_negated(l):
    return (l & 1) == 1
#replace last bit in the lit with 0(make it even)
def strip_lit(l):
    return l & ~1

#convert a variable aiger literal to the index that will be/are used in cudd
def lit_to_vIndex(l):
    if is_negated(l):
        logger.info('warning: cudding a negated literal')
    if l == error_fake_latch.lit:
        return errorVarIndex
    return l / 2 - 1

#convert a variable aiger literal to the index that will be/are used in cudd
def vIndex_to_lit(l):
    if l == errorVarIndex:
        return int(error_fake_latch.lit)
    return (l+1)*2

#always create an additional latch that confirm that a visited state was visited
def introduce_error_latch_if():
    global error_fake_latch
    global errorVarIndex

    if error_fake_latch:
        return

    error_fake_latch = aiger_symbol()
    #: :type: aiger_symbol
    error_symbol = get_err_symbol()

    error_fake_latch.lit = (int(spec.maxvar) + 1) * 2

    error_fake_latch.name = 'fake_error_latch'
    error_fake_latch.next = error_symbol.lit
    if spec.num_latches > 0:
        lastLatch = get_aiger_symbol(spec.latches, spec.num_latches - 1)
        errorVarIndex = lit_to_vIndex(lastLatch.lit) + 1
    else:
        errorVarIndex = spec.num_inputs

#A python generator function which returns a generator iterator.
#this function returns latches aiger symbols iterator
#aiger_symbol lit is the aiger index of the variable
def iterate_latches_and_error():
    introduce_error_latch_if()

    for i in range(int(spec.num_latches)):
        yield get_aiger_symbol(spec.latches, i)

    yield error_fake_latch


def parse_into_spec(aiger_file_name):
    global spec
    global firstMaxVar
    global first_ands_num
    logger.info('parsing..')
    #: :type: aiger
    spec = aiger_init() #initialiaze and return an aiger object
    #read aiger content and save them in the aiger object spec
    err = aiger_open_and_read_from_file(spec, aiger_file_name)
    firstMaxVar = spec.maxvar
    first_ands_num = spec.num_ands
    for i in range(int(spec.num_inputs)):
        input_aiger_symbol = get_aiger_symbol(spec.inputs, i)
        if (input_aiger_symbol.name.strip().startswith('controllable')):
            cont_input_lits.append(input_aiger_symbol.lit)
    assert not err, err

def get_lit_type(stripped_lit):
    if stripped_lit == error_fake_latch.lit:
        return None, error_fake_latch, None

    input_ = aiger_is_input(spec, stripped_lit)
    latch_ = aiger_is_latch(spec, stripped_lit)
    #need to be strip twice because of the implementation of the function
    #get_bdd_for_value(lit) the part that has to do with and gates
    and_ = aiger_is_and(spec, strip_lit(stripped_lit))

    return input_, latch_, and_

#maybe this should use some hashtable in order not to recompute anything.
#it returns the BDD for the aiger literal(aiger variable index)
def get_bdd_for_value(lit):  # lit is aiger variable index with sign
    global andgateTobdd
    stripped_lit = strip_lit(lit)
    if stripped_lit in andgateTobdd:
        if is_negated(lit):
            return ~andgateTobdd[stripped_lit]
        return andgateTobdd[stripped_lit]
    # we faked error latch and so we cannot call directly aiger_is_input,
    # aiger_is_latch, aiger_is_and
    input_, latch_, and_ = get_lit_type(stripped_lit)

    if stripped_lit == 0:
        res = cudd.Zero()

    elif input_ or latch_:
        res = cudd.IthVar(lit_to_vIndex(stripped_lit))

    elif and_:
        #: :type: aiger_and
        arg1 = get_bdd_for_value(int(and_.rhs0))
        arg2 = get_bdd_for_value(int(and_.rhs1))
        res = arg1 & arg2

    else:
        assert 0, 'should be impossible: if it is output then it is still either latch or and'

    andgateTobdd[stripped_lit] = res
    if is_negated(lit):
        res = ~res
    return res


def make_bdd_eq(value1, value2):
    return (value1 & value2) | (~value1 & ~value2)


def compose_transition_funcs():

    logger.info('compose_transition_funcs, nof_latches={0}...'
                .format(len(list(iterate_latches_and_error()))))

    #: :type: DdNode
    transition = cudd.One()
    for l in iterate_latches_and_error():
        #: :type: aiger_symbol
        l = l
        next_value_bdd = get_bdd_for_value(int(l.next))
        transition_funcs[l.lit] = next_value_bdd
    compose_substitution_array()
    andgateTobdd.clear()
    return transition

def compose_substitution_array():

    global substitutionDdArray
    if substitutionDdArray is not None:
        return
    logger.info('Compose substitution array')
    substitutionDdArray = pycudd.DdArray(
        spec.num_inputs + ((spec.num_latches + 1)))
    for i in range(int(spec.num_inputs)):
        substitutionDdArray[i] = cudd.IthVar(i)
    count = spec.num_inputs
    for l in iterate_latches_and_error():
        substitutionDdArray[count] = transition_funcs[l.lit]
            #substitutionDdArray.Push(transition_funcs[l.lit])
        count += 1


#bad is not used
def get_err_symbol():
    assert (spec.num_outputs == 1) ^ (spec.num_bad == 1), 'no safety properties'
    if spec.num_outputs == 1:
        return spec.outputs
    else:
        return spec.bad


def orBdds(bdd_1, bdd_2):
    return bdd_1.Or(bdd_2)


def andBdds(bdd_1, bdd_2):
    return bdd_1.And(bdd_2)


def get_cube(variables):
    #equivalent To:if not condition: raise AssertionError()
    assert len(variables)
    return reduce(andBdds, variables)


#returns a list of controllable/uncont. input BDDs
def _get_bdd_vars(filter_func):
    var_bdds = []
    for i in range(int(spec.num_inputs)):
        input_aiger_symbol = get_aiger_symbol(spec.inputs, i)
        if filter_func(input_aiger_symbol.name.strip()):
            out_var_bdd = get_bdd_for_value(input_aiger_symbol.lit)
            var_bdds.append(out_var_bdd)

    return var_bdds


#return a list of controllable variables bdds
def get_controllable_vars_bdds():
    #lambda expression
    return _get_bdd_vars(lambda name: name.startswith('controllable'))

#return a list of uncontrollable variables bdds
def get_uncontrollable_output_bdds():
    #lambda expression
    return _get_bdd_vars(lambda name: not name.startswith('controllable'))


def get_all_latches_as_bdds():
    #python list comprehension
    bdds = [get_bdd_for_value(l.lit) for l in iterate_latches_and_error()]
    return bdds

def compose_init_state_bdd():
    """ Initial state is 'all latches are zero' """
    logger.info('compose_init_state_bdd..')
    negatedLatches = map(lambda x: get_bdd_for_value(x.lit).Not(),
                         iterate_latches_and_error())
    init = reduce(andBdds, negatedLatches)
    return init

def vectorCompose(statesSet):
    return statesSet.VectorCompose(substitutionDdArray)

# compute controllable predecessor CPRE
def pre_sys_bdd(dst_states_bdd):

    out_vars_cube = get_cube(get_controllable_vars_bdds())
    in_vars_cube = get_cube(get_uncontrollable_output_bdds())
    preImage = vectorCompose(dst_states_bdd)
    exist_outs = preImage.ExistAbstract(out_vars_cube)  # ∃o VC(C)
    forall_inputs = exist_outs.UnivAbstract(in_vars_cube)  # ∀i ∃o VC(C)
    return forall_inputs

def calc_win_region(init_state_bdd, not_error_bdd):
    logger.info('calc_win_region..')
    iter_count = 0
    new_set_bdd = cudd.One()
    while True:

        iter_count += 1
        curr_set_bdd = new_set_bdd
        #optimization option here, merge with pre_sys_bdd
        #to use AndAbstract

        new_set_bdd = not_error_bdd & pre_sys_bdd(curr_set_bdd)

        if (new_set_bdd & init_state_bdd) == cudd.Zero():
            return cudd.Zero(), iter_count

        if new_set_bdd == curr_set_bdd:
            return new_set_bdd, iter_count

# compute uncontrollable predecessor UPRE
def UPRE_bdd(dst_states_bdd):
    controllable_vars_bdds = get_controllable_vars_bdds()
    uncontrollable_output_bdds = get_uncontrollable_output_bdds()
    out_vars_cube = get_cube(controllable_vars_bdds)
    in_vars_cube = get_cube(uncontrollable_output_bdds)
    preImage = vectorCompose(dst_states_bdd)
    forall_outs = preImage.UnivAbstract(out_vars_cube)  # ∀o VC(C)
    exist_inputs = forall_outs.ExistAbstract(in_vars_cube)  # ∃i ∀o VC(C)
    return exist_inputs


def calc_win_region_backward(init_state_bdd, error_bdd):
    logger.info('calc_win_region_backward..')
    iter_count = 0
    new_set_bdd = error_bdd
    while True:

        iter_count += 1
        curr_set_bdd = new_set_bdd
        new_set_bdd = UPRE_bdd(curr_set_bdd)
        if (new_set_bdd & init_state_bdd) != cudd.Zero():
            return cudd.Zero(), iter_count
        new_set_bdd = curr_set_bdd | new_set_bdd
        if new_set_bdd == curr_set_bdd:
            return ~new_set_bdd, iter_count


def get_nondet_strategy(win_region_bdd):
    logger.info('get_nondet_strategy..')
    return vectorCompose(win_region_bdd)

#bloem function
def optimizeCofactors(cbt, cbf):
    p = cbt & ~cbf
    n = ~cbt & cbf
    allVarBDDs = get_controllable_vars_bdds()
    allVarBDDs.extend(get_uncontrollable_output_bdds())
    allVarBDDs.extend(get_all_latches_as_bdds())
    for var in allVarBDDs:
        tempP = p.ExistAbstract(var)
        tempN = n.ExistAbstract(var)
        if tempP & tempN == cudd.Zero() or tempP & tempN == ~(cudd.One()):
            p = tempP
            n = tempN
    return p, n


def extract_output_funcs(non_det_strategy, init_state_bdd, withBloemOpt):
    """
    Calculate BDDs for output functions given a non-deterministic winning strategy.
    Cofactor-based approach.

    :return: dictionary ``controllable_variable_bdd -> func_bdd``
    """

    logger.info('extract_output_funcs..')

    output_models = dict()
    controls = get_controllable_vars_bdds()

    for c in get_controllable_vars_bdds():
        #logger.info('getting output function for ' + aiger_is_input(spec, vIndex_to_lit(c.NodeReadIndex())).name)
        # A set is an unordered collection with no duplicate elements.
        others = set(set(controls).difference({c}))
        if others:
            others_cube = get_cube(others)
            #: :type: DdNode
            c_arena = non_det_strategy.ExistAbstract(others_cube)
        else:
            c_arena = non_det_strategy

        # c_arena.PrintMinterm()

        can_be_true = c_arena.Cofactor(c)  # states (x,i) in which c can be true
        can_be_false = c_arena.Cofactor(~c)  # states (x,i) in which c can be true

        must_be_true = (~can_be_false) & can_be_true
        must_be_false = (~can_be_true) & can_be_false

        care_set = (must_be_true | must_be_false)
        if withBloemOpt:
            must_be_true, must_be_false = optimizeCofactors(can_be_true, can_be_false)

        care_set = (must_be_true | must_be_false)
        c_model = must_be_true.Restrict(care_set)

        #c_model = can_be_true.Restrict(care_set)

        #c_model = can_be_true.Restrict(care_set)
        output_models[c] = c_model
        non_det_strategy = non_det_strategy.Compose(c_model, c.NodeReadIndex())
    return output_models


def standard_synthesize(realiz_check, withBloemOpt):
    """ Calculate winning region and extract output functions.

    :return: - if realizable: <True, dictionary: controllable_variable_bdd -> func_bdd>
             - if not: <False, None>
    """
    logger.info('standard synthesize..')
    wr_size = 0
    init_state_bdd = compose_init_state_bdd()
    compose_transition_funcs()
    win_region, iter_count = calc_win_region_backward(
        init_state_bdd, get_bdd_for_value(error_fake_latch.lit))
    #for comparisons reasons
    if win_region == cudd.Zero():
        return False, None, iter_count, wr_size
    if realiz_check:
        return True, None, iter_count, wr_size
    # the +1 is for the error latch
    wr_size = win_region.CountMinterm(spec.num_latches + 1)
    #win_region.PrintMinterm()
    non_det_strategy = get_nondet_strategy(win_region)

    func_by_var = extract_output_funcs(non_det_strategy, init_state_bdd, withBloemOpt)
    #cudd.KillNode(non_det_strategy.__int__())
    #cudd.KillNode(win_region.__int__())
    transition_funcs.clear()
    return True, func_by_var, iter_count, wr_size


def negated(lit):
    return lit ^ 1


def next_lit():
    """ :return: next possible to add to the spec literal """
    return (int(spec.maxvar) + 1) * 2


def get_optimized_and_lit(a_lit, b_lit):
    if a_lit == 0 or b_lit == 0:
        return 0

    if a_lit == 1 and b_lit == 1:
        return 1

    if a_lit == 1:
        return b_lit

    if b_lit == 1:
        return a_lit

    if a_lit > 1 and b_lit > 1:
        a_b_lit = next_lit()
        aiger_add_and(spec, a_b_lit, a_lit, b_lit)
        return a_b_lit

    assert 0, 'impossible'


def walk(a_bdd):

    """
    Walk given BDD node (recursively).
    If given input BDD requires intermediate AND gates for its representation, the function adds them.
    Literal representing given input BDD is `not` added to the spec.

    :returns: literal representing input BDD
    :warning: variables in cudd nodes may be complemented, check with: ``node.IsComplement()``
    """
    global bddToAigCache
    #if(a_bdd in bddToAigCache):
    #    return bddToAigCache[a_bdd]
    #: :type: DdNode
    a_bdd = a_bdd
    if a_bdd.IsConstant():
        res = int(a_bdd == cudd.One())   # in aiger 0/1 = False/True
        return res
    # get an index of variable,
    # all variables used in bdds also introduced in aiger,
    # except fake error latch literal,
    # but fake error latch will not be used in output functions (at least we don't need this..)
    a_lit = vIndex_to_lit(a_bdd.NodeReadIndex())

    #assert a_lit != error_fake_latch.lit, 'using error latch in the definition of output function is not allowed'

    #: :type: DdNode
    t_bdd = a_bdd.T()
    #: :type: DdNode
    e_bdd = a_bdd.E()

    if t_bdd in bddToAigCache:
        t_lit = bddToAigCache[t_bdd]
    else:
        t_lit = walk(t_bdd)
        bddToAigCache[t_bdd] = t_lit
    if e_bdd in bddToAigCache:
        e_lit = bddToAigCache[e_bdd]
    else:
        e_lit = walk(e_bdd)
        bddToAigCache[e_bdd] = e_lit


    #t_lit = walk(t_bdd)
    #e_lit = walk(e_bdd)
    # ite(a_bdd, then_bdd, else_bdd)
    # = a*then + !a*else
    # = !(!(a*then) * !(!a*else))
    # -> in general case we need 3 more ANDs

    a_t_lit = get_optimized_and_lit(a_lit, t_lit)

    na_e_lit = get_optimized_and_lit(negated(a_lit), e_lit)

    n_a_t_lit = negated(a_t_lit)
    n_na_e_lit = negated(na_e_lit)

    ite_lit = get_optimized_and_lit(n_a_t_lit, n_na_e_lit)

    res = negated(ite_lit)
    if a_bdd.IsComplement():
        res = negated(res)
    return res


def model_to_aiger(c_bdd, func_bdd, introduce_output):
    """ Update aiger spec with a definition of ``c_bdd``
    """
    #: :type: DdNode
    c_bdd = c_bdd
    c_lit = vIndex_to_lit(c_bdd.NodeReadIndex())
    func_as_aiger_lit = walk(func_bdd)
    aiger_redefine_input_as_and(spec, c_lit, func_as_aiger_lit, func_as_aiger_lit)
    if introduce_output:
        aiger_add_output(spec, c_lit, '')


def init_cudd():
    global cudd
    cudd = pycudd.DdManager()
    cudd.SetDefault()
    cudd.AutodynEnable(4)

#############Lazy Synthesis################Lazy Synthesis#######################
#############Lazy Synthesis################Lazy Synthesis#######################
#############Lazy Synthesis################Lazy Synthesis#######################
#############Lazy Synthesis################Lazy Synthesis#######################

def getPreimage(statesSet, constraints, inputCube):
    if(constraints is None):
        return vectorCompose(statesSet).ExistAbstract(inputCube)
    return vectorCompose(statesSet).AndAbstract(constraints, inputCube)

def modelCheck(constraints, initCube):
    logger.info('model checking..')
    levels = []
    if constraints is None:
        constraints = cudd.One()
    out_vars_cube = get_cube(get_controllable_vars_bdds())
    in_vars_cube = get_cube(get_uncontrollable_output_bdds())
    inputCube = out_vars_cube & in_vars_cube
    #we start from the set of unsafe states
    errorBdd = get_bdd_for_value(error_fake_latch.lit)
    visitedStates = errorBdd
    levels.append(errorBdd)
    initCheck = None  # to check if initial point is in a level
    while True:
        # compute newTmp
        currentLevel = visitedStates
        preImage = getPreimage(currentLevel, constraints, inputCube)
        newStates = preImage & ~visitedStates
        #newStates.PrintMinterm()
        # check if initial point is in the computed level
        initCheck = (newStates & initCube)
        if initCheck != ~cudd.One() and initCheck != cudd.Zero():
            levels.insert(0, newStates)
            logger.info('model checking finished!')
            return False, levels
        # check if fixed point
        if(newStates == cudd.Zero() or newStates == ~cudd.One()):
            return True, visitedStates
        visitedStates = visitedStates | newStates
        levels.insert(0, newStates)
        # end of check if fixed point

#The below function mix lazy and standard methods
#i.e. at every step it first make one UPRE computation
#and add it to error BDD from there do the model checking
def mixedmodelCheck(constraints, initCube):
    global accumErrorBDD
    logger.info('model checking..')
    levels = []
    if constraints is None:
        constraints = cudd.One()
    out_vars_cube = get_cube(get_controllable_vars_bdds())
    in_vars_cube = get_cube(get_uncontrollable_output_bdds())
    inputCube = out_vars_cube & in_vars_cube
    #we start from the set of unsafe states
    if accumErrorBDD is None:
        accumErrorBDD = get_bdd_for_value(error_fake_latch.lit)
    else:
        accumErrorBDD = accumErrorBDD | UPRE_bdd(accumErrorBDD)
    visitedStates = accumErrorBDD
    levels.append(accumErrorBDD)
    initCheck = None  # to check if initial point is in a level
    while True:
        # compute newTmp
        currentLevel = visitedStates
        preImage = getPreimage(currentLevel, constraints, inputCube)
        newStates = preImage & ~visitedStates
        #newStates.PrintMinterm()
        # check if initial point is in the computed level
        initCheck = (newStates & initCube)
        if initCheck != ~cudd.One() and initCheck != cudd.Zero():
            levels.insert(0, newStates)
            logger.info('model checking finished!')
            return False, levels
        # check if fixed point
        if(newStates == cudd.Zero() or newStates == ~cudd.One()):
            return True, visitedStates
        visitedStates = visitedStates | newStates
        levels.insert(0, newStates)
        # end of check if fixed point


#The below function merges two levels lists
    #levelList1 and levelList2 are lists of BDDs
def mergeLevels(list1, list2):
    return list(map(lambda x, y: (cudd.Zero() if x is None else x)
    | (cudd.Zero() if y is None else y), list1, list2))

def getTime():
    return int(round(time.time() * 1000))

def checkForFailure(lvls, constraints, stIndex, initCube, inputCube):
    logger.info('Checking failure')
    if constraints is None:
        constraints = cudd.One()
    init = initCube
    prevLvl = copy.deepcopy(lvls[stIndex])
    #It is enough to check level 1
    #the above is incorrect(check only level 1) because once a level is
    #totally empty we can not remove any transitions from the levels
    #that are above it
    i = stIndex
    while i > 0:
        #print i
        lvl = prevLvl
        lvlPrimed = lvl
        oldPreImage = vectorCompose(lvlPrimed)
        newPreImage = oldPreImage & constraints
        newPreImage = newPreImage.AndAbstract(lvls[i - 1], inputCube)
        if(newPreImage == cudd.Zero() or newPreImage == ~cudd.One()):
            return False
        if(i == 1):
            initCheck = newPreImage & init
            if (initCheck == cudd.Zero() or initCheck == ~cudd.One()):
                return False
            else:
                return True
        prevLvl = newPreImage
        i = i - 1
    return False


    #idea 1: when computing nxtLevelPreImg outPreImg the first level keep
    #the results than to compute the outPreImg of the next level
    #the new nxtLevelPreImg can be computed by
    #nxtLevelPreImg = old nextLevelPreimage - the exact L_i+1 preimage
    #outPreImg = old outPreImg union the exact L_i+1 preimage
def FixTransitionRelationWithoutErrorPathsWithJumps(generalDelete, withBreak,
     initCube, contrCube, unContrCube, mixed):

    logger.info('Fixing Transition Relation.........')
    errorSets = dict()
    iterCount = 0
    #for amba general delete is better in tim
    #but worst in terms of WR size
    #general delete means do not restric deletion to current level
    levels = []  # levels copy to work on
    keeplvls = []  # levels copy not to touch to be used in all iterations
    mcLvls = []  # levels from only one modelchecking function call
    constraints = None
    sep = -1  # shortest error path
    #the two below params are used for jumps
    goingUp = False
    avSetStart = -1
    inputCube = contrCube & unContrCube
    while True:
        goToNextIteration = False
        if mixed :
            isCorrect, lvlsOrLR = mixedmodelCheck(constraints, initCube)
        else:
            isCorrect, lvlsOrLR = modelCheck(constraints, initCube)
        if(isCorrect):
            return constraints, ~lvlsOrLR # in this case lvlsOrLR is the loosing region
        constraints = None  # reset constraints as modelcheck failed
        mcLvls = lvlsOrLR  # levels obtained from model check
        keeplvls = mergeLevels(keeplvls, mcLvls)  # merge levels from all iters
        levels = copy.deepcopy(keeplvls)  # keepLevels no toch
        sep = len(mcLvls)
        errorSets[sep - 1] = (mcLvls[sep - 1] | errorSets[sep - 1])\
         if ((sep - 1) in errorSets) else mcLvls[sep - 1]
        sep = len(keeplvls)
        print '******************************'
        print((
        'iteration: ' + str(iterCount) + ' sep: ' + str(sep)
        ))
        print '******************************'
        startTime = getTime()
        #reset jumping
        goingUp = False
        avSetStart = -1
        i = 0
        while i >= 0 and i < (sep - 1):
            print('------------------------------')
            print(('it: ' + str(iterCount) + ', i: ' + str(i)))
            if not generalDelete:
                workLevel = levels[i]
            #remove error states from a level
            if((i in errorSets)):
                if(levels[i] == errorSets[i]):
                    print 'all are errors'
                    failure = checkForFailure(levels,
                         constraints, 1, initCube, inputCube)
                    if(failure):
                        return cudd.Zero(), cudd.Zero() #Failure
                    else:
                        goToNextIteration = True
                if not generalDelete:
                    workLevel = levels[i] & ~(errorSets[i])
            #computing to be removed transitions
            logger.info('computing to be removed transitions....')
            nxtlvlPrimed = reduce(orBdds, levels[i + 1:])
            nxtLevelPreImg = vectorCompose(nxtlvlPrimed)

            logger.info('outPreImg computation...')
            #outPreImg = (vectorCompose(~nxtlvlPrimed) & (cudd.One() if constraints is None else constraints)).ExistAbstract(contrCube)
            outPreImg = (~nxtLevelPreImg).AndAbstract(
            (cudd.One() if constraints is None else constraints), contrCube)

            #outPreImg = ((~nxtLevelPreImg) & (cudd.One() if constraints is None else constraints)).ExistAbstract(contrCube)
            if not generalDelete:
                part1 = outPreImg & workLevel
            else:
                #part1 = self.AND(outPreImg, self.NOT(self.errorsBDD))
                part1 = outPreImg
            ToReTr = nxtLevelPreImg & part1  # To be removed Trans
            logger.info('Accumulating ToReTr....')
            #Remove these transitions from TRFct(Add them to constraints)
            if((not ToReTr == cudd.Zero()) and (not ToReTr == ~cudd.One())):
                if(constraints is None):
                    constraints = ~ToReTr
                else:
                    constraints = constraints & ~ToReTr

            # computing the avoid set
            logger.info('computing the avoid set....')
            AvSet = part1.UnivAbstract(unContrCube)
            if generalDelete:
                AvSet = AvSet & levels[i]# after testing this can be removed and it still works but checkForFailure can't be from 1'
            if((not AvSet == cudd.Zero()) and (not AvSet == ~cudd.One())):
                if not goingUp and i > 0:
                    goingUp = True
                    avSetStart = i
                levels[i] = levels[i] & ~AvSet
                if(levels[i] == cudd.Zero() or levels[i] == ~cudd.One() or (i == 0 and (levels[i] & initCube == cudd.Zero() or levels[i] & initCube == ~cudd.One()))):
                    goToNextIteration = True
                    #with break it works but very expensive and with generalDelete it fails
                    if(withBreak):
                        break
                if(i > 0):
                    print(('Go up from: ' + str(i)))
                    i = i - 2
            else:
                if(goingUp):
                    goingUp = False
                    i = avSetStart
            i = i + 1
        if(not goToNextIteration):
            failure = checkForFailure(levels, constraints, 1, initCube, inputCube)
            if(failure):
                return cudd.Zero(), cudd.Zero() #Failure
        endTime = getTime()
        print(("iteration finished in: " + str(endTime - startTime)))
        print('##############################')
        iterCount += 1


def ComputeToReTrAndAvset(currLvlIndx, levels, globalNextPre, goingUp):
    logger.info('lowerLevelsPreImg...')
    i = currLvlIndx
    if i == 0 and not goingUp:
        nxtlvls = reduce(orBdds, levels[1:])
        lowerLevelsPreImg = vectorCompose(nxtlvls)
    else:  # if i > 0 or i ==0 and going up
        if(not goingUp):
            currLvlPreImg = vectorCompose(levels[i] & (~(reduce(orBdds, levels[i+1:]))))
            lowerLevelsPreImg = globalNextPre & ~currLvlPreImg
        else:  # if going up
            nextLvl = levels[i+1] # & ~avset
            nextLvlPreImg = vectorCompose(nextLvl)
            lowerLevelsPreImg = globalNextPre | nextLvlPreImg
    return lowerLevelsPreImg



#an idea to fix the problem is to create a dict for avsets so that we can imitate
#the exact computation of the original method
def OptFixTransitionRelationWithoutErrorPathsWithJumps(generalDelete, withBreak,
     initCube, contrCube, unContrCube, debugMode, mixed):
    global shortestPath  # Resillient Guarantee
    maxSep = 0
    shortestPath = 0  # Resillient Guarantee
    original_shortestPath = 0  # Resillient Guarantee
    lowerLevelsPreImg = None
    lowerLevelsPreImgFixed = None
    logger.info('Fixing Transition Relation.........')
    errorSets = dict()
    iterCount = 0
    #for amba general delete is better in tim
    #but worst in terms of WR size
    #general delete means do not restric deletion to current level
    levels = []  # levels copy to work on
    keeplvls = []  # levels copy not to touch to be used in all iterations
    mcLvls = []  # levels from only one modelchecking function call
    constraints = None
    sep = -1  # shortest error path
    #the two below params are used for jumps
    goingUp = False
    avSetStart = -1
    inputCube = contrCube & unContrCube
    workLevel = None
    while True:
        goToNextIteration = False
        if mixed :
            isCorrect, lvlsOrLR = mixedmodelCheck(constraints, initCube)
        else:
            isCorrect, lvlsOrLR = modelCheck(constraints, initCube)
        if(isCorrect):
            return constraints, ~lvlsOrLR, iterCount, maxSep
            # in this case lvlsOrLR is the loosing region


        constraints = None  # reset constraints as modelcheck failed
        AvSet = cudd.Zero()
        mcLvls = lvlsOrLR  # levels obtained from model check
        keeplvls = mergeLevels(keeplvls, mcLvls)  # merge levels from all iters
        levels = copy.deepcopy(keeplvls)  # keepLevels no toch
        sep = len(mcLvls)
        if iterCount == 0:  # Resillient Guarantee
            original_shortestPath = sep - 1  # Resillient Guarantee
        shortestPath = original_shortestPath  # Resillient Guarantee
        maxi = 0  # Resillient Guarantee
        errorSets[sep - 1] = (mcLvls[sep - 1] | errorSets[sep - 1])\
         if ((sep - 1) in errorSets) else mcLvls[sep - 1]
        sep = len(keeplvls)
        if debugMode:
            print '******************************'
            print((
            'iteration: ' + str(iterCount) + ' sep: ' + str(sep)
            ))
            print '******************************'
            startTime = getTime()
        #reset jumping
        goingUp = False
        avSetStart = -1
        i = 0
        while i >= 0 and i < (sep - 1):
            maxi = max(maxi,i)  # Resillient Guarantee
            if debugMode:
                print('------------------------------')
                print(('it: ' + str(iterCount) + ', i: ' + str(i)))
            if not generalDelete:
                workLevel = levels[i]
            #remove error states from a level
            if((i in errorSets)):
                if(levels[i] == errorSets[i]):
                    if debugMode:
                        print 'all are errors'
                    failure = checkForFailure(levels,
                         constraints, 1, initCube, inputCube)
                    if(failure):
                        return cudd.Zero(), cudd.Zero(), iterCount, maxSep
                    else:
                        goToNextIteration = True
                if not generalDelete:
                    workLevel = levels[i] & ~(errorSets[i])
            #computing to be removed transitions
            lowerLevelsPreImg = ComputeToReTrAndAvset(
                i, levels, lowerLevelsPreImg, goingUp)
            #logger.info('outPreImg computation...')
            if constraints is None:
                outsidePreImg = (~lowerLevelsPreImg).ExistAbstract(contrCube)
                if not generalDelete:
                    part1 = outsidePreImg & workLevel
                else:
                    part1 = outsidePreImg
            else:
                if not generalDelete:
                    outsidePreImg = (~lowerLevelsPreImg).AndAbstract(
                    (constraints & workLevel), contrCube)
                else:
                    outsidePreImg = (~lowerLevelsPreImg).AndAbstract(
                    (constraints), contrCube)
                part1 = outsidePreImg
            #if not generalDelete:
            #    part1 = outsidePreImg & workLevel
            #else:
            #    #part1 = self.AND(outPreImg, self.NOT(self.errorsBDD))
                part1 = outsidePreImg
            ToReTr = lowerLevelsPreImg & part1  # To be removed Trans
            # computing the avoid set
            #logger.info('computing the avoid set....')
            AvSet = part1.UnivAbstract(unContrCube)
            #logger.info('accumulate constraints....')
            #Add ToReTr to constraints
            if((not ToReTr == cudd.Zero()) and (not ToReTr == ~cudd.One())):
                if(constraints is None):
                    constraints = ~ToReTr
                else:
                    constraints = constraints & ~ToReTr

            if generalDelete:
                AvSet = AvSet & levels[i]# after testing this can be removed and it still works but checkForFailure can't be from 1'

            if((not AvSet == cudd.Zero()) and (not AvSet == ~cudd.One())):
                if not goingUp and i > 0:
                    lowerLevelsPreImgFixed = copy.deepcopy(lowerLevelsPreImg)
                    goingUp = True
                    avSetStart = i
                levels[i] = levels[i] & ~AvSet
                if(levels[i] == cudd.Zero() or levels[i] == ~cudd.One() or (i == 0 and (levels[i] & initCube == cudd.Zero() or levels[i] & initCube == ~cudd.One()))):
                    goToNextIteration = True
                    #with break it works but very expensive and with generalDelete it fails
                    if(withBreak):
                        break
                if(i > 0):
                    i = i - 2
                    #print(('Go up from: ' + str(i)))
            else:
                if(goingUp):
                    lowerLevelsPreImg = lowerLevelsPreImgFixed
                    goingUp = False
                    i = avSetStart
            i = i + 1
        if(not goToNextIteration):
            failure = checkForFailure(levels, constraints, 1, initCube, inputCube)
            if(failure):
                return cudd.Zero(), cudd.Zero(), iterCount, maxSep  # Failure
        if debugMode:
            endTime = getTime()
            print(("iteration finished in: " + str(endTime - startTime)))
            print('##############################')
        iterCount += 1
        shortestPath = shortestPath - maxi -1  # Resillient Guarantee, as i represents the max level from which i can avoid the later levels, the -1 is for the error latch
        maxSep = max(maxSep, sep)

def lazy_Synthesize(generalDelete, withBreak, realiz_check, standardStrategy, noOptimization, debug, withBloemOpt, mixed):
    logger.info('lazy synthesize..')
    iter_count = 0
    max_sep = 0
    wr_size = 0
    initBdd = compose_init_state_bdd()
    compose_transition_funcs()
    controllable_vars_bdds = get_controllable_vars_bdds()
    uncontrollable_output_bdds = get_uncontrollable_output_bdds()
    contrCube = get_cube(controllable_vars_bdds)
    unContrCube = get_cube(uncontrollable_output_bdds)
    #constraints can be seen also as non-deterministic strategy
    #as they represent valid constraints
    if noOptimization:
        constraints, winRegion = FixTransitionRelationWithoutErrorPathsWithJumps(
            generalDelete, withBreak, initBdd, contrCube, unContrCube, mixed)
    else:
        constraints, winRegion, iter_count, max_sep = OptFixTransitionRelationWithoutErrorPathsWithJumps(
            generalDelete, withBreak, initBdd, contrCube, unContrCube, debug, mixed)
    #for comparisons reasons

    if winRegion == cudd.Zero():
        return False, None, wr_size, iter_count, max_sep
    if realiz_check:
        return True, None, wr_size, iter_count, max_sep
    #the +1 is for the error latch
    wr_size = winRegion.CountMinterm(spec.num_latches + 1)
    if standardStrategy:
        non_det_strategy = get_nondet_strategy(winRegion)
    else:
        non_det_strategy = constraints

    func_by_var = extract_output_funcs(non_det_strategy, initBdd, withBloemOpt)
    #cudd.KillNode(non_det_strategy.__int__())
    #cudd.KillNode(winRegion.__int__())
    transition_funcs.clear()
    return True, func_by_var, wr_size, iter_count, max_sep

###########################End of LazySynthesis########End of LazySynthesis#####
###########################End of LazySynthesis########End of LazySynthesis#####
###########################End of LazySynthesis########End of LazySynthesis#####
###########################End of LazySynthesis########End of LazySynthesis#####
###########################End of LazySynthesis########End of LazySynthesis#####
###########################End of LazySynthesis########End of LazySynthesis#####

def aiger_write_without_specification(aiger, out_file_name,onFile, comments):
    logger.info('write aag without specification..')
    aiger = aiger
    #create temp file with only controllable inputs'andgates and minimize it
    filename, file_extension = os.path.splitext(out_file_name)
    temp_file_name = filename+'_temp' + file_extension
    fileOb = open(temp_file_name, 'w+')
    fileOb.write('aag ' + str(spec.maxvar) + ' ' + str(spec.num_inputs + spec.num_latches) + ' 0 ' + str(len(cont_input_lits)) + ' ' + str((spec.maxvar - (firstMaxVar) + len(cont_input_lits))) + os.linesep)
    #uncontrollable inputs
    for i in range(int(spec.num_inputs)):
        input_aiger_symbol = get_aiger_symbol(spec.inputs, i)
        fileOb.write(str(input_aiger_symbol.lit) + os.linesep)
    #latches as inputs
    for i in range(int(spec.num_latches)):
        input_aiger_symbol = get_aiger_symbol(spec.latches, i)
        fileOb.write(str(input_aiger_symbol.lit) + os.linesep)
    #controllable inputs as oputput
    for c_in in cont_input_lits:
        fileOb.write(str(c_in) + os.linesep)
    #and gates
    for i in range(int(spec.num_ands)):
        andGate = get_aiger_and(spec.ands, i)
        if(andGate.lhs > (firstMaxVar * 2) or andGate.lhs in cont_input_lits):
            fileOb.write(str(andGate.lhs) + ' ' + str(andGate.rhs0) + ' '
            + str(andGate.rhs1) + os.linesep)
    fileOb.close()
    #return
    ABCminimization(temp_file_name)
    tempAiger = aiger_init()  # initialiaze and return an aiger object
    #read aiger content and save them in the aiger object spec
    aiger_open_and_read_from_file(tempAiger, temp_file_name)



    #Get final result result result result result result result result result
    result = []
    inputNames = []
    latchesNames = []
    outputName = None
    delayedAndGates = []
    #first line
    andGatesNb = first_ands_num + tempAiger.num_ands

    #uncontrollable inputs
    count = 0
    for i in range(int(spec.num_inputs)):
        in_sym = get_aiger_symbol(spec.inputs, i)
        result.append(str(in_sym.lit))
        inputNames.append('i'+ str(count) + ' ' + in_sym.name)
        count += 1
    #latches
    for i in range(int(spec.num_latches)):
        ltch_sym = get_aiger_symbol(spec.latches, i)
        result.append(str(ltch_sym.lit) + ' ' + str(ltch_sym.next))
        latchesNames.append('l'+ str(i) + ' ' + ltch_sym.name)
    #output
    out_sym = get_aiger_symbol(spec.outputs, 0)
    result.append(str(out_sym.lit))
    if out_sym is not None and out_sym.name is not None:
        outputName = 'o0 ' + out_sym.name
    else:
        outputName = 'o0 o0'
    #original and gates
    for i in range(int(spec.num_ands)):
        andGate = get_aiger_and(spec.ands, i)
        if andGate.lhs <= (firstMaxVar * 2) and andGate.lhs not in cont_input_lits:
            result.append(str(andGate.lhs) + ' '
            + str(andGate.rhs0) + ' ' + str(andGate.rhs1))
    #temp original uncont map
    tou = dict()
    for i in range(int(spec.num_inputs)):  # spec contains only uncont inputs
        temp_in_sym = get_aiger_symbol(tempAiger.inputs, i)
        in_sym = get_aiger_symbol(spec.inputs, i)
        tou[temp_in_sym.lit] = in_sym.lit
    #temp to original latch
    tol = dict()
    stIndex = int(spec.num_inputs)
    for i in range(int(spec.num_latches)):
        temp_in_sym = get_aiger_symbol(tempAiger.inputs, i + stIndex)
        ltch_sym = get_aiger_symbol(spec.latches, i)
        tol[temp_in_sym.lit] = ltch_sym.lit
    #temp original cont map
    toc = dict()
    toc_backup = dict()  #fix
    for i in range(len(cont_input_lits)):
        temp_out_sym = get_aiger_symbol(tempAiger.outputs, i)
        strippedLit = strip_lit(temp_out_sym.lit)
        if temp_out_sym.lit <= 1:
            delayedAndGates.append(str(cont_input_lits[i]) + ' '
            + str(temp_out_sym.lit) + ' ' + str(temp_out_sym.lit))
        elif strippedLit in tol:
            if temp_out_sym.lit % 2 == 0:
                delayedAndGates.append(str(cont_input_lits[i]) + ' '
            + str(tol[strippedLit]) + ' ' + str(tol[strippedLit]))
            else:
                delayedAndGates.append(str(cont_input_lits[i]) + ' '
            + str(tol[strippedLit] + 1) + ' ' + str(tol[strippedLit] + 1))
        elif strippedLit in tou:
            if temp_out_sym.lit % 2 == 0:
                delayedAndGates.append(str(cont_input_lits[i]) + ' '
            + str(tou[strippedLit]) + ' ' + str(tou[strippedLit]))
            else:
                delayedAndGates.append(str(cont_input_lits[i]) + ' '
            + str(tou[strippedLit] + 1) + ' ' + str(tou[strippedLit] + 1))
        else:
            if temp_out_sym.lit in toc:  #fix
                toc_backup[temp_out_sym.lit] = cont_input_lits[i]  #fix
            else:  #fix
                toc[temp_out_sym.lit] = cont_input_lits[i]
    #temp new and gates
    tna = dict()
    nextAgIndex = firstMaxVar * 2 + 2
    for i in range(int(tempAiger.num_ands)):
        andGate = get_aiger_and(tempAiger.ands, i)
        #print andGate.lhs, andGate.rhs0, andGate.rhs1
        nlhs = 0
        if andGate.lhs in toc:
            nlhs = toc[andGate.lhs]
        else:
            tna[andGate.lhs] = nextAgIndex
            nlhs = nextAgIndex
            nextAgIndex += 2
        rhs0Neg = False
        rhs1Neg = False
        rhs0 = andGate.rhs0
        rhs1 = andGate.rhs1
        if rhs0 % 2 == 1:
            rhs0Neg = True
            rhs0 -= 1
        if rhs1 % 2 == 1:
            rhs1Neg = True
            rhs1 -= 1
        nrhs0 = 0
        nrhs1 = 0
        if(rhs0 in tou):
            nrhs0 = tou[rhs0]
        elif(rhs0 in toc):
            nrhs0 = toc[rhs0]
        elif(rhs0 in tol):
            nrhs0 = tol[rhs0]
        elif(rhs0 in tna):
            nrhs0 = tna[rhs0]
        if(rhs1 in tou):
            nrhs1 = tou[rhs1]
        elif(rhs1 in toc):
            nrhs1 = toc[rhs1]
        elif(rhs1 in tol):
            nrhs1 = tol[rhs1]
        elif(rhs1 in tna):
            nrhs1 = tna[rhs1]
        if rhs0Neg:
            nrhs0 += 1
        if rhs1Neg:
            nrhs1 += 1
        result.append(str(nlhs) + ' ' + str(nrhs0) + ' ' + str(nrhs1))
        if andGate.lhs in toc_backup:  #fix
            result.append(str(toc_backup[andGate.lhs]) + ' ' + str(nrhs0) + ' ' + str(nrhs1))  #fix
            andGatesNb += 1  #fix
        #if the output was negated
        if (andGate.lhs + 1) in toc:
            result.append(str(toc[andGate.lhs + 1]) + ' ' + str(nlhs + 1) + ' ' + str(nlhs + 1))
            andGatesNb += 1

        #print nlhs, nrhs0, nrhs1
    andGatesNb += len(delayedAndGates)
    result.extend(delayedAndGates)
    result.extend(inputNames)
    result.extend(latchesNames)
    result.append(outputName)
    comments.append('solution size: ' + str(tempAiger.num_ands))
    result.extend(comments)
    #first line
    result.insert(0, 'aag '
    + str(firstMaxVar + andGatesNb - len(cont_input_lits) - first_ands_num ) + ' '
    + str(spec.num_inputs) + ' ' + str(spec.num_latches) + ' 1 '
    + str(andGatesNb))
    if onFile:
        sol_file_name = filename + file_extension
        fileOb = open(sol_file_name, 'w+')
        for i in range(len(result)):
            fileOb.write(result[i] + os.linesep)
        fileOb.close()
    else:
        for i in range(len(result)):
            print result[i]





def main(aiger_file_name, out_file_name, output_full_circuit, realiz_check,
     lazy, genDel, nobreak, ctrStrtgy, noOpt, debug, noblOpt, mix):
    global cont_input_lits
    sys.setrecursionlimit(100000)
    iter_count = 0
    wr_size = 0
    max_sep = 0
    if not debug:
        logger.disabled = True
    """ Open aiger file, synthesize the circuit and write the result to output file.

    :returns: boolean value 'is realizable?'
    """
    #BDD manager initialization
    init_cudd()
    #read aiger content and save them in the aiger object "spec
    parse_into_spec(aiger_file_name)
    start_time = getTime()
    #call Synthesizer
    if lazy:
        #we should also try cut strategy...not vc(loosingregion) \land wr
        realizable, func_by_var, wr_size, iter_count, max_sep = lazy_Synthesize(genDel, not nobreak, realiz_check, not ctrStrtgy, noOpt, debug, not noblOpt, mix)
    else:
        realizable, func_by_var, iter_count, wr_size = standard_synthesize(realiz_check, not noblOpt)

    if realiz_check:
        return realizable

    if realizable:
        #for fct in list(transition_funcs.values()):
        #    cudd.KillNode(fct.__int__())
        cudd.ReduceHeap(4, 1)
        logger.info('model to aiger...')
        for (c_bdd, func_bdd) in func_by_var.items():
            model_to_aiger(c_bdd, func_bdd, output_full_circuit)
        end_time = getTime()
        #adding comments to end results
        aiger_comments = []
        aiger_comments.append('c')
        aiger_comments.append('time: ' + str(end_time - start_time) + ' ms')
        aiger_comments.append('iterations #: ' + str(iter_count))
        if(lazy):
            aiger_comments.append('max error path #: ' + str(max_sep))
            if not nobreak:  # Resillient Guarantee
                aiger_comments.append('Resillient Guarantee #: ' + str(shortestPath))  # Resillient Guarantee
        aiger_comments.append('WR size: ' + str(wr_size))

        #aiger_reencode(spec)  # some model checkers do not like unordered variable names (when e.g. latch is > add)
        logger.info('Write results...')
        if out_file_name:
            aiger_write_without_specification(spec, out_file_name, True,
                aiger_comments)
            #aiger_open_and_write_to_file(spec, out_file_name)
            #ABCminimization(out_file_name)
        else:
            aiger_write_without_specification(spec, aiger_file_name, False,
                aiger_comments)
            #print 'false out file name'
            #res, string = aiger_write_to_string(spec, aiger_ascii_mode, 268435456)
            #assert res != 0 or out_file_name is None, 'writing failure'
            #logger.info('\n' + string)
        return True
    print "UNREALIZABLE"
    return False


def exit_if_status_request(args):
    if args.status:
        print('-'*80)
        print('The current status of development')
        print('-- ' + '\n-- '.join(status))
        print('-'*80)
        exit(0)
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aiger Format Based Simple Synthesizer')
    parser.add_argument('aiger', metavar='aiger', type=str, nargs='?', default=None, help='input specification in AIGER format')
    parser.add_argument('--out', '-o', metavar='out', type=str, required=False, default=None,
                        help='output file in AIGER format (if realizable)')
    parser.add_argument('--full', action='store_true', default=False,
                        help='produce a full circuit that has outputs other than error bit')
    parser.add_argument('--status', '-s', action='store_true', default=False,
                        help='Print current status of development')

    parser.add_argument('--realizability', '-r', action='store_true', default=False,
                        help='Check Realizability only (do not produce circuits)')

    parser.add_argument('--lazy', '-lazy', action='store_true', default=False,
            help='Lazy synthesis, default is standard method')
    parser.add_argument('--genDel', '-genDel', action='store_true', default=False,
            help='general delete when an escape is found')
    parser.add_argument('--nobreak', '-nobreak', action='store_true', default=False,
            help='When an escape is found for init state do not stop the algorithm')
    parser.add_argument('--ctrStrtgy', '-ctrStrtgy', action='store_true', default=False,
            help='compute output functions from Constraints instead of from winning strategies')
    parser.add_argument('--noOpt', '-noOpt', action='store_true', default=False,
            help='If we do not want to use the optimization for computing escapes')
    parser.add_argument('--debug', '-debug', action='store_true', default=False,
            help='debug mode enable logger')
    parser.add_argument('--noblOpt', '-noblOpt', action='store_true', default=False,
            help='do not apply bloem optimization when extracting solutions')
    parser.add_argument('--mix', '-mix', action='store_true', default=False,
            help='mix lazy and standard')

    args = parser.parse_args()

    exit_if_status_request(args)

    if not args.aiger:
        print('aiger file is required, exit')
        exit(-1)

    setup_logging(0)


    is_realizable = main(args.aiger, args.out, args.full, args.realizability,args.lazy, args.genDel, args.nobreak, args.ctrStrtgy, args.noOpt, args.debug, args.noblOpt, args.mix)

    logger.info(['unrealizable', 'realizable'][is_realizable])

    exit([EXIT_STATUS_UNREALIZABLE, EXIT_STATUS_REALIZABLE][is_realizable])
