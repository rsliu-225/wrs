from collections import deque
from copy import deepcopy


class PTreeNode:
    def __init__(self, node_id=-1, children=None):
        if children is None:
            children = []
        self.id = node_id
        self.children = children


class PTree:
    def __init__(self, num):
        self.root = PTreeNode()
        self._build_permutation_tree(self.root, list(range(num)))
        self.max_depth = num

    def _build_permutation_tree(self, node, available_child_ids):
        # clean up the node's children
        node.children = []
        for i, node_id in enumerate(available_child_ids):
            child_node = PTreeNode(node_id)
            node.children.append(child_node)
            child_available = [cid for ci, cid in enumerate(available_child_ids) if ci != i]
            self._build_permutation_tree(child_node, child_available)

    def prune(self, seq=None):
        # prune the subtree by failed sequence
        if seq is None:
            seq = []

        curr = self.root
        curr_idx = 0
        prev = None
        for node_id in seq:
            children_ids = [child.id for child in curr.children]
            if node_id not in children_ids:
                print("The sequence not exist or already pruned")
                return
            prev = curr
            curr_idx = children_ids.index(node_id)
            curr = curr.children[curr_idx]

        # delete failed sequence as well as its sub trees
        del prev.children[curr_idx]

    def output(self, seq=None):
        # output the remained leaves whose depth equals to max_depth
        if seq is None:
            seq = []

        result = []

        # if given a sequence, go to the end of seq first
        curr = self.root
        curr_depth = 0
        for i, node_id in enumerate(seq):
            children_ids = [node.id for node in curr.children]
            if node_id not in children_ids:
                print(f"The sequence is not valid at {seq[i]} in {seq}")
                return result
            curr = curr.children[children_ids.index(node_id)]
            curr_depth += 1

        # use BFS to traversal the tree
        queue = deque([(node, seq[:curr_depth]) for node in curr.children])
        curr_depth += 1

        while queue:
            q_len = len(queue)
            for _ in range(q_len):
                node, prev_node_ids = queue.popleft()
                curr_node_ids = deepcopy(prev_node_ids)
                curr_node_ids.append(node.id)
                if curr_depth == self.max_depth:
                    result.append(curr_node_ids)
                else:
                    for child_node in node.children:
                        queue.append((child_node, curr_node_ids))
            curr_depth += 1

        return result

    def output_sgl(self, seq=None):
        # output the remained leaves whose depth equals to max_depth
        if seq is None:
            seq = []

        result = []

        # if given a sequence, go to the end of seq first
        curr = self.root
        curr_depth = 0
        for i, node_id in enumerate(seq):
            children_ids = [node.id for node in curr.children]
            if node_id not in children_ids:
                print(f"The sequence is not valid at {seq[i]} in {seq}")
                return result
            curr = curr.children[children_ids.index(node_id)]
            curr_depth += 1

        # use BFS to traversal the tree
        queue = deque([(node, seq[:curr_depth]) for node in curr.children])
        curr_depth += 1

        while queue:
            q_len = len(queue)
            for _ in range(q_len):
                node, prev_node_ids = queue.popleft()
                curr_node_ids = deepcopy(prev_node_ids)
                curr_node_ids.append(node.id)
                if curr_depth == self.max_depth:
                    return curr_node_ids
                else:
                    for child_node in node.children:
                        queue.append((child_node, curr_node_ids))
            curr_depth += 1

        return result

if __name__ == '__main__':
    x = PTree(5)
    print(x.output_sgl())
    x.prune([0, 1])
    print(x.output_sgl())
    x.prune([0, 2])
    print(x.output_sgl(seq=[0, 2]))
