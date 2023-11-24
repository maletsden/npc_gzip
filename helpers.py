from __future__ import annotations
from collections import Counter
import re
import numba


def infix_to_rpn(expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    functions = {'r', 'i', 'a', 'c', 'g', 't', 'n', 'q', 'f'}

    output = []
    stack = []

    def handle_operator(token):
        while (stack and stack[-1] != '(' and
               (precedence.get(stack[-1], 0) >= precedence[token])):
            output.append(stack.pop())
        stack.append(token)

    for token in re.findall(r'\w+|\S', expression):
        if token.isnumeric():
            output.append(token)
        elif token in precedence:
            handle_operator(token)
        elif token in functions:
            stack.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # Remove the '(' from the stack
            if stack and stack[-1] in functions:
                output.append(stack.pop())
        else:
            output.append(token)  # Variable or function argument

    while stack:
        output.append(stack.pop())

    return ''.join(output)


class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right

    def __str__(self):
        return self.left, self.right


def huffman_code_tree(node, binString=''):
    '''
    Function to find Huffman Code
    '''
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, binString + '0'))
    d.update(huffman_code_tree(r, binString + '1'))
    return d


def make_tree(nodes):
    '''
    Function to make tree
    :param nodes: Nodes
    :return: Root of the tree
    '''
    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    return nodes[0][0]


def huffman_encode(input: str) -> str:
    freq = dict(Counter(input))
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    node = make_tree(freq)
    encoding = huffman_code_tree(node)
    # for i in encoding:
    #    print(f'{i} : {encoding[i]}')
    return ''.join([encoding[c] for c in input])


@numba.njit()
def encode_rle(message) -> tuple[list, list]:
    encoded_message = ""
    i = 0
    encoded_value = []
    encode_count = []
    while (i <= len(message) - 1):
        count = 1
        ch = message[i]
        j = i
        while (j < len(message) - 1):
            if (message[j] == message[j + 1]):
                count = count + 1
                j = j + 1
            else:
                break
        encoded_value.append(ch)
        encode_count.append(count)
        # encoded_message=encoded_message+str(count)+ch
        i = j + 1
    return encoded_value, encode_count


class unique_element:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1


def findCombinationsUtil(arr, index, num, reducedNum, found_combinations):
    # Base condition
    if (reducedNum < 0):
        return

    # If combination is
    # found, print it
    if (reducedNum == 0):
        found_combinations += list(perm_unique(arr[:index]))
        # for i in range(index):
        #     print(arr[i], end = " ")
        # print("")
        return

    # Find the previous number stored in arr[].
    # It helps in maintaining increasing order
    prev = 1 if (index == 0) else arr[index - 1]

    # note loop starts from previous
    # number i.e. at array location
    # index - 1
    for k in range(prev, num + 1):
        # next element of array is k
        arr[index] = k

        # call recursively with
        # reduced number
        findCombinationsUtil(arr, index + 1, num,
                             reducedNum - k, found_combinations)


def findCombinations(n):
    # array to store the combinations
    # It can contain max n elements
    arr = [0] * n
    found_combinations = []

    # find all combinations
    findCombinationsUtil(arr, 0, n, n, found_combinations)
    return found_combinations